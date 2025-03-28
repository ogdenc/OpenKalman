/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Library routines for native Eigen types.
 */

#ifndef OPENKALMAN_EIGEN_LIBRARY_INTERFACE_HPP
#define OPENKALMAN_EIGEN_LIBRARY_INTERFACE_HPP

#include <type_traits>
#include <tuple>
#include <random>
#include "linear-algebra/interfaces/eigen/details/eigen-forward-declarations.hpp"

namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<Eigen3::eigen_general<true> T>
  struct library_interface<T>
#else
  template<typename T>
  struct library_interface<T, std::enable_if_t<Eigen3::eigen_general<T, true>>>
#endif
  {
    template<typename Derived>
    using LibraryBase = Eigen3::EigenAdapterBase<Derived,
      std::conditional_t<Eigen3::eigen_array_general<T>, Eigen::ArrayBase<Derived>,
      std::conditional_t<Eigen3::eigen_matrix_general<T>, Eigen::MatrixBase<Derived>, Eigen::EigenBase<Derived>>>>;

  private:

    template<typename Arg, std::size_t...I, typename...Ind>
    static constexpr void
    check_index_bounds(const Arg& arg, std::index_sequence<I...>, Ind...ind)
    {
      ([](auto max, auto ind){ if (ind < 0 or ind >= max) throw std::out_of_range {
        ("Index " + std::to_string(I) + " is out of bounds: it is " + std::to_string(ind) +
        " but should be in range [0..." + std::to_string(max - 1) + "].")};
      }(get_index_dimension_of<I>(arg), ind),...);
    }


    template<typename Arg>
    static constexpr decltype(auto)
    get_coeff(Arg&& arg, Eigen::Index i, Eigen::Index j)
    {
      //// If we did range checking, this is what it would look like:
      //check_index_bounds(arg, std::make_index_sequence<2> {}, i, j);
      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (Eigen3::eigen_DiagonalMatrix<Arg> or Eigen3::eigen_DiagonalWrapper<Arg>)
      {
        if (i == j)
          return static_cast<Scalar>(get_coeff(nested_object(std::forward<Arg>(arg)), i, 0));
        else
          return static_cast<Scalar>(0);
      }
      else if constexpr (Eigen3::eigen_TriangularView<Arg>)
      {
        constexpr int Mode {std::decay_t<Arg>::Mode};
        if ((i > j and (Mode & int{Eigen::Upper}) != 0x0) or (i < j and (Mode & int{Eigen::Lower}) != 0x0))
          return static_cast<Scalar>(0);
        else
          return static_cast<Scalar>(get_coeff(nested_object(std::forward<Arg>(arg)), i, j));
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        return get_coeff(nested_object(std::forward<Arg>(arg)), i, j);
      }
      else
      {
        auto evaluator = Eigen::internal::evaluator<std::decay_t<Arg>>(arg);
        constexpr bool use_coeffRef = (Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0 and
          std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>>;
        if constexpr (use_coeffRef) return evaluator.coeffRef(i, j);
        else return evaluator.coeff(i, j);
      }
    }


    template<typename Indices>
    static constexpr std::tuple<Eigen::Index, Eigen::Index>
    extract_indices(const Indices& indices)
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      auto it = ranges::begin(indices);
      auto e = ranges::end(indices);
      if (it == e) return {0, 0};
      auto i = static_cast<std::size_t>(*it);
      if (++it == e) return {i, 0};
      auto j = static_cast<std::size_t>(*it);
      if (++it == e) return {i, j};
      throw std::logic_error("Wrong number of indices on component access");
    }

  public:

#ifdef __cpp_lib_ranges
  template<typename Arg, std::ranges::input_range Indices> requires 
    std::convertible_to<std::ranges::range_value_t<Indices>, const typename std::decay_t<Arg>::Index> and
    (collections::size_of_v<Indices> == dynamic_size or collections::size_of_v<Indices> <= 2)
  static constexpr value::scalar decltype(auto)
#else
  template<typename Arg, typename Indices, std::enable_if_t<
    (collections::size_of_v<Indices> == dynamic_size or collections::size_of_v<Indices> <= 2), int> = 0>
  static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      using Scalar = scalar_type_of_t<Arg>;
      auto [i, j] = extract_indices(indices);
      if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        constexpr int Mode {std::decay_t<Arg>::Mode};
        bool transp = (i > j and (Mode & int{Eigen::Upper}) != 0) or (i < j and (Mode & int{Eigen::Lower}) != 0);
        if constexpr (value::complex<Scalar>)
        {
          if (transp) return static_cast<Scalar>(value::conj(get_coeff(std::as_const(arg), j, i)));
          else return static_cast<Scalar>(get_coeff(std::as_const(arg), i, j));
        }
        else
        {
          if (transp) return get_coeff(std::forward<Arg>(arg), j, i);
          else return get_coeff(std::forward<Arg>(arg), i, j);
        }
      }
      else
      {
        return get_coeff(std::forward<Arg>(arg), i, j);
      }
    }


#ifdef __cpp_lib_ranges
    template<typename Arg, std::ranges::input_range Indices> requires (not std::is_const_v<Arg>) and
      std::convertible_to<std::ranges::range_value_t<Indices>, const typename Arg::Index> and
      (collections::size_of_v<Indices> == dynamic_size or collections::size_of_v<Indices> <= 2) and
      std::assignable_from<decltype(get_coeff(std::declval<Arg&>(), 0, 0)), const scalar_type_of_t<Arg>&> and
      ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0) and
      (not Eigen3::eigen_DiagonalWrapper<Arg>) and (not Eigen3::eigen_TriangularView<Arg>)
#else
    template<typename Arg, typename Indices, std::enable_if_t<(not std::is_const_v<Arg>) and
      (collections::size_of_v<Indices> == dynamic_size or collections::size_of_v<Indices> <= 2) and
      std::is_assignable<decltype(get_coeff(std::declval<Arg&>(), 0, 0)), const scalar_type_of_t<Arg>&>::value and
      (Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0 and
      (not Eigen3::eigen_DiagonalWrapper<Arg>) and (not Eigen3::eigen_TriangularView<Arg>), int> = 0>
#endif
    static void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      using Scalar = scalar_type_of_t<Arg>;
      auto [i, j] = extract_indices(indices);
      if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        constexpr int Mode {std::decay_t<Arg>::Mode};
        bool transp = (i > j and (Mode & int{Eigen::Upper}) != 0) or (i < j and (Mode & int{Eigen::Lower}) != 0);
        if constexpr (value::complex<Scalar>)
        {
          if (transp) get_coeff(arg, j, i) = value::conj(s);
          else get_coeff(arg, i, j) = s;
        }
        else
        {
          if (transp) get_coeff(arg, j, i) = s;
          else get_coeff(arg, i, j) = s;
        }
      }
      else
      {
        get_coeff(arg, i, j) = s;
      }
    }

  private:

    template<typename Arg>
    static decltype(auto)
    wrap_if_nests_by_reference(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_general<Arg>)
      {
        constexpr auto Flags = Eigen::internal::traits<std::remove_reference_t<Arg>>::Flags;
        if constexpr (std::is_lvalue_reference_v<Arg> and static_cast<bool>(Flags & Eigen::NestByRefBit))
          return std::forward<Arg>(arg);
        else
          return Eigen3::make_eigen_wrapper(std::forward<Arg>(arg));
      }
      else
      {
        return Eigen3::make_eigen_wrapper(std::forward<Arg>(arg));
      }
    }

  public:

    template<typename Arg>
    static decltype(auto)
    to_native_matrix(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        return std::forward<Arg>(arg);
      }
      else if constexpr (internal::library_wrapper<Arg>)
      {
        return to_native_matrix(OpenKalman::nested_object(std::forward<Arg>(arg)));
      }
      else if constexpr (Eigen3::eigen_ArrayWrapper<Arg>)
      {
        return to_native_matrix(nested_object(std::forward<Arg>(arg)));
      }
      else if constexpr (Eigen3::eigen_array_general<Arg>)
      {
        return wrap_if_nests_by_reference(std::forward<Arg>(arg)).matrix();
      }
      else if constexpr (Eigen3::eigen_matrix_general<Arg>)
      {
        return wrap_if_nests_by_reference(std::forward<Arg>(arg));
      }
      else if constexpr (not Eigen3::eigen_general<Arg> and directly_accessible<Arg> and std::is_lvalue_reference_v<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        constexpr int rows = dynamic_dimension<Arg, 0> ? Eigen::Dynamic : index_dimension_of_v<Arg, 0>;
        constexpr int cols = dynamic_dimension<Arg, 1> ? Eigen::Dynamic : index_dimension_of_v<Arg, 1>;

        if constexpr (layout_of_v<Arg> == Layout::stride)
        {
          auto [s0, s1] = internal::strides(arg);
          using S0 = decltype(s0);
          using S1 = decltype(s1);
          constexpr int es0 = []() -> int {
            if constexpr (value::fixed<S0>) return static_cast<std::ptrdiff_t>(S0{});
            else return Eigen::Dynamic;
          }();
          constexpr int es1 = []() -> int {
            if constexpr (value::fixed<S1>) return static_cast<std::ptrdiff_t>(S1{});
            else return Eigen::Dynamic;
          }();
          using IndexType = typename std::decay_t<Arg>::Index;
          auto is0 = static_cast<IndexType>(static_cast<std::ptrdiff_t>(s0));
          auto is1 = static_cast<IndexType>(static_cast<std::ptrdiff_t>(s1));

          if constexpr (value::fixed<S0> and
            (es0 == 1 or (value::fixed<S1> and es0 < es1)))
          {
            using M = Eigen::Matrix<Scalar, rows, cols, Eigen::ColMajor>;
            Eigen::Stride<es1, es0> strides {is1, is0};
            return Eigen::Map<const M, Eigen::Unaligned, decltype(strides)> {internal::raw_data(arg), rows, cols, strides};
          }
          else if constexpr (value::fixed<S1> and
            (es1 == 1 or (value::fixed<S0> and es1 < es0)))
          {
            using M = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;
            Eigen::Stride<es0, es1> strides {is0, is1};
            return Eigen::Map<const M, Eigen::Unaligned, decltype(strides)> {internal::raw_data(arg), rows, cols, strides};
          }
          else
          {
            if (is1 < is0)
            {
              using M = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;
              Eigen::Stride<es0, es1> strides {is0, is1};
              return Eigen::Map<const M, Eigen::Unaligned, decltype(strides)> {internal::raw_data(arg), rows, cols, strides};
            }
            else
            {
              using M = Eigen::Matrix<Scalar, rows, cols, Eigen::ColMajor>;
              Eigen::Stride<es1, es0> strides {is1, is0};
              return Eigen::Map<const M, Eigen::Unaligned, decltype(strides)> {internal::raw_data(arg), rows, cols, strides};
            }
          }
        }
        else
        {
          constexpr auto l = layout_of_v<Arg> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor;
          using M = Eigen::Matrix<Scalar, rows, cols, l>;
          return Eigen::Map<const M> {internal::raw_data(arg), rows, cols};
        }
      }
      else
      {
        return Eigen3::make_eigen_wrapper(std::forward<Arg>(arg));
      }
    }

  private:

    template<typename Scalar, int rows, int cols, auto options>
    using dense_type = std::conditional_t<Eigen3::eigen_array_general<T, true>,
      Eigen::Array<Scalar, rows, cols, options>, Eigen::Matrix<Scalar, rows, cols, options>>;

    template<typename Scalar, std::size_t rows, std::size_t cols, auto options>
    using writable_type = dense_type<Scalar,
      (rows == dynamic_size ? Eigen::Dynamic : static_cast<int>(rows)),
      (cols == dynamic_size ? Eigen::Dynamic : static_cast<int>(cols)), options>;

  public:

#ifdef __cpp_concepts
    template<typename To, Eigen3::eigen_general From> requires (std::assignable_from<To&, From&&>)
#else
    template<typename To, typename From, std::enable_if_t<Eigen3::eigen_general<From> and std::is_assignable_v<To&, From&&>, int> = 0>
#endif
    static void assign(To& a, From&& b)
    {
      if constexpr (Eigen3::eigen_DiagonalWrapper<From>)
      {
        if constexpr (vector<nested_object_of_t<From>>)
          a = std::forward<From>(b);
        else
          a = diagonal_of(std::forward<From>(b)).asDiagonal();
      }
      else
      {
        a = std::forward<From>(b);
      }
    }

  private:

    template<typename Descriptors>
    static constexpr decltype(auto)
    extract_descriptors(Descriptors&& descriptors)
    {
      if constexpr (pattern_tuple<Descriptors>)
      {
        constexpr auto dim = std::tuple_size_v<std::decay_t<Descriptors>>;
        static_assert(dim <= 2);
        if constexpr (dim == 0) return std::tuple {coordinate::Axis{}, coordinate::Axis{}};
        else if constexpr (dim == 1) return std::tuple {std::get<0>(std::forward<Descriptors>(descriptors)), coordinate::Axis{}};
        else return std::forward<Descriptors>(descriptors);
      }
      else
      {
#ifdef __cpp_lib_ranges
        namespace ranges = std::ranges;
#endif
        auto it = ranges::begin(descriptors);
        auto e = ranges::end(descriptors);
        if (it == e) return std::tuple {coordinate::Axis{}, coordinate::Axis{}};
        auto i = *it;
        if (++it == e) return std::tuple {i, coordinate::Axis{}};
        auto j = *it;
        if (++it == e) return std::tuple {i, j};
        throw std::logic_error("Wrong number of vector space descriptors");
      }
    }

  public:

#ifdef __cpp_concepts
    template<Layout layout, value::number Scalar, coordinate::euclidean_pattern_collection Ds>
#else
    template<Layout layout, typename Scalar, typename Ds, std::enable_if_t<value::number<Scalar> and
      coordinate::euclidean_pattern_collection<Ds>, int> = 0>
#endif
    static auto make_default(Ds&& ds)
    {
      using IndexType = typename Eigen::Index;

      if constexpr (pattern_tuple<Ds> and collections::size_of_v<Ds> > 2)
      {
        constexpr auto options = layout == Layout::right ? Eigen::RowMajor : Eigen::ColMajor;

        if constexpr (fixed_pattern_tuple<Ds>)
        {
          auto sizes = std::apply([](auto&&...d){
            return Eigen::Sizes<static_cast<std::ptrdiff_t>(coordinate::size_of_v<decltype(d)>)...> {};
          }, std::forward<Ds>(ds));

          return Eigen::TensorFixedSize<Scalar, decltype(sizes), options, IndexType> {};
        }
        else
        {
          return std::apply([](auto&&...d){
            using Ten = Eigen::Tensor<Scalar, collections::size_of_v<Ds>, options, IndexType>;
            return Ten {static_cast<IndexType>(get_size(d))...};
            }, std::forward<Ds>(ds));
        }
      }
      else
      {
        auto [d0, d1] = extract_descriptors(std::forward<Ds>(ds));
        constexpr auto dim0 = coordinate::size_of_v<decltype(d0)>;
        constexpr auto dim1 = coordinate::size_of_v<decltype(d1)>;

        static_assert(layout != Layout::right or dim0 == 1 or dim1 != 1,
          "Eigen does not allow creation of a row-major column vector.");
        static_assert(layout != Layout::left or dim0 != 1 or dim1 == 1,
          "Eigen does not allow creation of a column-major row vector.");

        constexpr auto options =
          layout == Layout::right or (layout == Layout::none and dim0 == 1 and dim1 != 1) ?
          Eigen::RowMajor : Eigen::ColMajor;

        using M = writable_type<Scalar, dim0, dim1, options>;

        if constexpr (dim0 == dynamic_size or dim1 == dynamic_size)
          return M {static_cast<IndexType>(get_size(d0)), static_cast<IndexType>(get_size(d1))};
        else
          return M {};
      }


    }


#ifdef __cpp_concepts
    template<Layout layout, writable Arg, std::convertible_to<scalar_type_of_t<Arg>> S, std::convertible_to<scalar_type_of_t<Arg>>...Ss>
      requires (layout == Layout::right) or (layout == Layout::left)
#else
    template<Layout layout, typename Arg, typename S, typename...Ss, std::enable_if_t<writable<Arg> and
      (layout == Layout::right or layout == Layout::left) and
      std::conjunction<std::is_convertible<S, typename scalar_type_of<Arg>::type>,
          std::is_convertible<Ss, typename scalar_type_of<Arg>::type>...>::value, int> = 0>
#endif
    static void fill_components(Arg& arg, const S s, const Ss ... ss)
    {
      if constexpr (layout == Layout::left) ((arg.transpose() << s), ... , ss);
      else ((arg << s), ... , ss);
    }


#ifdef __cpp_lib_ranges
    template<value::dynamic C, coordinate::euclidean_pattern_collection Ds> requires
      (collections::size_of_v<Ds> != dynamic_size) and (collections::size_of_v<Ds> <= 2)
    static constexpr constant_matrix auto
#else
    template<typename C, typename Ds, std::enable_if_t<value::dynamic<C> and
      coordinate::euclidean_pattern_collection<Ds> and
      (collections::size_of_v<Ds> != dynamic_size) and (collections::size_of_v<Ds> <= 2)), int> = 0>
    static constexpr auto
#endif
    make_constant(C&& c, Ds&& ds)
    {
      auto [d0, d1] = extract_descriptors(std::forward<Ds>(ds));
      constexpr auto dim0 = coordinate::size_of_v<decltype(d0)>;
      constexpr auto dim1 = coordinate::size_of_v<decltype(d1)>;

      auto value = value::to_number(std::forward<C>(c));
      using Scalar = std::decay_t<decltype(value)>;
      constexpr auto options = (dim0 == 1 and dim1 != 1) ? Eigen::RowMajor : Eigen::ColMajor;
      using M = writable_type<Scalar, dim0, dim1, options>;

      using IndexType = typename Eigen::Index;

      if constexpr (fixed_pattern_collection<Ds>)
        return M::Constant(value);
      else
        return M::Constant(static_cast<IndexType>(dim0), static_cast<IndexType>(get_size(d1)), value);
      // Note: We don't address orders greater than 2 because Eigen::Tensor's constant has substantial limitations.
    }


#ifdef __cpp_concepts
    template<typename Scalar, coordinate::euclidean_pattern_collection Ds> requires
      (collections::size_of_v<Ds> != dynamic_size) and (collections::size_of_v<Ds> <= 2)
    static constexpr identity_matrix auto
#else
    template<typename Scalar, typename...Ds, std::enable_if_t<coordinate::euclidean_pattern_collection<Ds> and
      (collections::size_of_v<Ds> != dynamic_size) and (collections::size_of_v<Ds> <= 2), int> = 0>
    static constexpr auto
#endif
    make_identity_matrix(Ds&& ds)
    {
      auto [d0, d1] = extract_descriptors(std::forward<Ds>(ds));
      constexpr auto dim0 = coordinate::size_of_v<decltype(d0)>;
      constexpr auto dim1 = coordinate::size_of_v<decltype(d1)>;

      constexpr auto options = (dim0 == 1 and dim1 != 1) ? Eigen::RowMajor : Eigen::ColMajor;
      using M = writable_type<Scalar, dim0, dim1, options>;

      using IndexType = typename Eigen::Index;

      if constexpr (fixed_pattern_collection<Ds>)
        return M::Identity();
      else
        return M::Identity(static_cast<IndexType>(get_size(d0)), static_cast<IndexType>(get_size(d1)));
    }


#ifdef __cpp_concepts
    template<TriangleType t, Eigen3::eigen_dense_general Arg> requires std::is_lvalue_reference_v<Arg> or
      (not std::is_lvalue_reference_v<typename Eigen::internal::ref_selector<std::decay_t<Arg>>::non_const_type>)
#else
    template<TriangleType t, typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg> and (std::is_lvalue_reference_v<Arg> or
      not std::is_lvalue_reference_v<typename Eigen::internal::ref_selector<std::remove_reference_t<Arg>>::non_const_type>), int> = 0>
#endif
    static constexpr auto
    make_triangular_matrix(Arg&& arg)
    {
      constexpr auto Mode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
      return arg.template triangularView<Mode>();
    }


#ifdef __cpp_concepts
    template<HermitianAdapterType t, Eigen3::eigen_dense_general Arg> requires std::is_lvalue_reference_v<Arg>
#else
    template<HermitianAdapterType t, typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg> and std::is_lvalue_reference_v<Arg>, int> = 0>
#endif
    static constexpr auto
    make_hermitian_adapter(Arg&& arg)
    {
      constexpr auto Mode = t == HermitianAdapterType::upper ? Eigen::Upper : Eigen::Lower;
      return arg.template selfadjointView<Mode>();
    }


    // to_euclidean not defined--rely on default

    // from_euclidean not defined--rely on default

    // wrap_angles not defined--rely on default


#ifdef __cpp_concepts
    template<typename Arg, typename...Begin, typename...Size> requires (sizeof...(Begin) <= 2)
#else
    template<typename Arg, typename...Begin, typename...Size, std::enable_if_t<(sizeof...(Begin) <= 2), int> = 0>
#endif
    static auto
    get_slice(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size)
    {
      auto b0 = [](const auto& begin){
        using Begin0 = std::decay_t<decltype(std::get<0>(begin))>;
        if constexpr (value::fixed<Begin0>) return std::integral_constant<Eigen::Index, Begin0::value>{};
        else return static_cast<Eigen::Index>(std::get<0>(begin));
      }(begin);

      auto b1 = [](const auto& begin){
        if constexpr (sizeof...(Begin) < 2) return std::integral_constant<Eigen::Index, 0>{};
        else
        {
          using Begin1 = std::decay_t<decltype(std::get<1>(begin))>;
          if constexpr (value::fixed<Begin1>) return std::integral_constant<Eigen::Index, Begin1::value>{};
          else return static_cast<Eigen::Index>(std::get<1>(begin));
        }
      }(begin);

      auto s0 = [](const auto& size){
        using Size0 = std::decay_t<decltype(std::get<0>(size))>;
        if constexpr (value::fixed<Size0>) return std::integral_constant<Eigen::Index, Size0::value>{};
        else return static_cast<Eigen::Index>(std::get<0>(size));
      }(size);

      auto s1 = [](const auto& size){
        if constexpr (sizeof...(Size) < 2) return std::integral_constant<Eigen::Index, 1>{};
        else
        {
          using Size1 = std::decay_t<decltype(std::get<1>(size))>;
          if constexpr (value::fixed<Size1>) return std::integral_constant<Eigen::Index, Size1::value>{};
          else return static_cast<Eigen::Index>(std::get<1>(size));
        }
      }(size);

      constexpr int S0 = static_cast<int>(value::fixed<decltype(s0)> ? static_cast<Eigen::Index>(s0) : Eigen::Dynamic);
      constexpr int S1 = static_cast<int>(value::fixed<decltype(s1)> ? static_cast<Eigen::Index>(s1) : Eigen::Dynamic);

      decltype(auto) m = to_native_matrix(std::forward<Arg>(arg));
      using M = decltype(m);

      constexpr auto Flags = Eigen::internal::traits<std::remove_reference_t<M>>::Flags;

      if constexpr (directly_accessible<M> and not (std::is_lvalue_reference_v<M> and static_cast<bool>(Flags & Eigen::NestByRefBit)))
      {
        // workaround for Eigen::Block's special treatment of directly accessible nested types:
        auto rep {std::forward<M>(m).template replicate<1,1>()};
        using B = Eigen::Block<const decltype(rep), S0, S1>;
        if constexpr ((value::fixed<Size> and ...))
          return B {std::move(rep), std::move(b0), std::move(b1)};
        else
          return B {std::move(rep), std::move(b0), std::move(b1), std::move(s0), std::move(s1)};
      }
      else
      {
        using M_noref = std::remove_reference_t<M>;
        using XprType = std::conditional_t<std::is_lvalue_reference_v<M>, M_noref, const M_noref>;
        using B = Eigen::Block<XprType, S0, S1>;
        if constexpr ((value::fixed<Size> and ...))
          return B {std::forward<M>(m), std::move(b0), std::move(b1)};
        else
          return B {std::forward<M>(m), std::move(b0), std::move(b1), std::move(s0), std::move(s1)};
      }
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_dense_general Arg, Eigen3::eigen_general Block, typename...Begin> requires (sizeof...(Begin) <= 2)
#else
    template<typename Arg, typename Block, typename...Begin, std::enable_if_t<
      Eigen3::eigen_dense_general<Arg> and Eigen3::eigen_general<Block> and (sizeof...(Begin) <= 2), int> = 0>
#endif
    static void
    set_slice(Arg& arg, Block&& block, const Begin&...begin)
    {
      auto [b0, b1] = [](Eigen::Index bs0, Eigen::Index bs1, const auto&...){ return std::tuple {bs0, bs1}; }
        (static_cast<std::size_t>(begin)..., 0_uz, 0_uz);

      if constexpr (Eigen3::eigen_Block<Block>)
      {
        if (std::addressof(arg) == std::addressof(block.nestedExpression()) and b0 == block.startRow() and b1 == block.startCol())
          return;
      }

      constexpr auto Bx0 = static_cast<int>(index_dimension_of_v<Block, 0>);
      constexpr auto Bx1 = static_cast<int>(index_dimension_of_v<Block, 1>);
      using Bk = Eigen::Block<std::remove_reference_t<Arg>, Bx0, Bx1>;

      if constexpr (not has_dynamic_dimensions<Block>)
      {
        Bk {arg, b0, b1} = std::forward<Block>(block);
      }
      else
      {
        auto s0 = static_cast<Eigen::Index>(get_index_dimension_of<0>(block));
        auto s1 = static_cast<Eigen::Index>(get_index_dimension_of<1>(block));
        Bk {arg, b0, b1, s0, s1} = std::forward<Block>(block);
      }
    }


#ifdef __cpp_concepts
    template<TriangleType t, Eigen3::eigen_SelfAdjointView A, Eigen3::eigen_general B> requires (not hermitian_matrix<A>) and
      set_triangle_defined_for<T, t, decltype(OpenKalman::nested_object(std::declval<A>())), B&&>
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      Eigen3::eigen_general<B> and Eigen3::eigen_SelfAdjointView<A> and (not hermitian_matrix<A>) and
      set_triangle_defined_for<T, t, decltype(OpenKalman::nested_object(std::declval<A>())), B&&>, int> = 0>
#endif
    static void
    set_triangle(A&& a, B&& b)
    {
      // A SelfAdjointView won't always be a hermitian_adapter
      if constexpr ((t == TriangleType::lower and interface::indexible_object_traits<T>::hermitian_adapter_type == HermitianAdapterType::upper) or
          (t == TriangleType::upper and interface::indexible_object_traits<T>::hermitian_adapter_type == HermitianAdapterType::lower))
        internal::set_triangle<t>(OpenKalman::nested_object(std::forward<A>(a)), OpenKalman::adjoint(std::forward<B>(b)));
      else
        internal::set_triangle<t>(OpenKalman::nested_object(std::forward<A>(a)), std::forward<B>(b));
    }


#ifdef __cpp_concepts
    template<TriangleType t, typename A, Eigen3::eigen_general B> requires
      (diagonal_adapter<A> or t == TriangleType::diagonal) and
      std::assignable_from<decltype(OpenKalman::diagonal_of(std::declval<A&&>())), decltype(OpenKalman::diagonal_of(std::declval<B&&>()))>
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      Eigen3::eigen_general<B> and (diagonal_adapter<A> or t == TriangleType::diagonal) and
      std::is_assignable_v<decltype(OpenKalman::diagonal_of(std::declval<A&&>())), decltype(OpenKalman::diagonal_of(std::declval<B&&>()))>, int> = 0>
#endif
    static void
    set_triangle(A&& a, B&& b)
    {
      assign(OpenKalman::diagonal_of(std::forward<A>(a)), OpenKalman::diagonal_of(std::forward<B>(b)));
    }


#ifdef __cpp_concepts
    template<TriangleType t, typename A, Eigen3::eigen_general B> requires
      (Eigen3::eigen_MatrixWrapper<A> or Eigen3::eigen_ArrayWrapper<A>) and
      set_triangle_defined_for<T, t, decltype(OpenKalman::nested_object(std::declval<A>())), B&&>
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<Eigen3::eigen_general<B> and
      (Eigen3::eigen_MatrixWrapper<A> or Eigen3::eigen_ArrayWrapper<A>) and
      set_triangle_defined_for<T, t, decltype(OpenKalman::nested_object(std::declval<A>())), B&&>, int> = 0>
#endif
    static void
    set_triangle(A&& a, B&& b)
    {
      internal::set_triangle<t>(OpenKalman::nested_object(std::forward<A>(a)), std::forward<B>(b));
    }


#ifdef __cpp_concepts
    template<TriangleType t, Eigen3::eigen_dense_general A, Eigen3::eigen_general B> requires
      (not Eigen3::eigen_MatrixWrapper<A>) and (not Eigen3::eigen_ArrayWrapper<A>) and
      writable<A&&> and (t != TriangleType::diagonal)
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      Eigen3::eigen_dense_general<A> and Eigen3::eigen_general<B> and
      (not Eigen3::eigen_MatrixWrapper<A>) and (not Eigen3::eigen_ArrayWrapper<A>) and
      writable<A&&> and (t != TriangleType::diagonal), int> = 0>
#endif
    static void
    set_triangle(A&& a, B&& b)
    {
      a.template triangularView<t == TriangleType::upper ? Eigen::Upper : Eigen::Lower>() = std::forward<B>(b);
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_dense_general Arg> requires square_shaped<Arg> or dimension_size_of_index_is<Arg, 0, 1> or
      std::is_lvalue_reference_v<Arg> or (not std::is_lvalue_reference_v<typename std::decay_t<Arg>::Nested>)
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg> and
      (square_shaped<Arg> or dimension_size_of_index_is<Arg, 0, 1> or std::is_lvalue_reference_v<Arg> or
        not std::is_lvalue_reference_v<typename std::decay_t<Arg>::Nested>), int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      if constexpr (not vector<Arg>) if (not is_vector(arg)) throw std::invalid_argument {
        "Argument of to_diagonal must have 1 column; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};

      if constexpr (square_shaped<Arg> or dimension_size_of_index_is<Arg, 0, 1>)
      {
        return internal::make_fixed_size_adapter(std::forward<Arg>(arg)); // Make one-dimensional matrix
      }
      else if constexpr (Eigen3::eigen_array_general<Arg>)
      {
        return arg.matrix().asDiagonal();
      }
      else
      {
        return arg.asDiagonal();
      }
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_SelfAdjointView Arg> requires (hermitian_adapter_type_of_v<Arg> == HermitianAdapterType::lower)
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_SelfAdjointView<Arg>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, a "lower SelfAdjointView wrapper is irrelevant
      return OpenKalman::to_diagonal(nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_TriangularView Arg> requires (triangle_type_of_v<Arg> == TriangleType::lower)
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_TriangularView<Arg> and (triangle_type_of_v<Arg> == TriangleType::lower), int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, a "lower" TriangularView wrapper is irrelevant
      return OpenKalman::to_diagonal(nested_object(std::forward<Arg>(arg)));
    }


    template<typename Arg>
#ifdef __cpp_concepts
    static constexpr vector decltype(auto)
#else
    static constexpr decltype(auto)
#endif
    diagonal_of(Arg&& arg)
    {
      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (Eigen3::eigen_DiagonalWrapper<Arg>)
      {
        using Diag = decltype(nested_object(std::forward<Arg>(arg))); //< must be nested_object(...) rather than .diagonal() because of const_cast
        using EigenTraits = Eigen::internal::traits<std::decay_t<Diag>>;
        constexpr auto rows = EigenTraits::RowsAtCompileTime;
        constexpr auto cols = EigenTraits::ColsAtCompileTime;

        static_assert(cols != 1, "For Eigen::DiagonalWrapper<T> interface, T should never be a column vector "
                                 "because diagonal_of function handles this case.");
        if constexpr (cols == 0)
        {
          auto ret {nested_object(std::forward<Arg>(arg))};
          return ret;
        }
        else if constexpr (rows == 1 or rows == 0)
        {
          auto ret {OpenKalman::transpose(nested_object(std::forward<Arg>(arg)))};
          return ret;
        }
        else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
        {
          decltype(auto) diag = nested_object(std::forward<Arg>(arg));
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          return M {M::Map(to_dense_object(std::forward<Diag>(diag)).data(),
            get_index_dimension_of<0>(diag) * get_index_dimension_of<1>(diag))};
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          return M {M::Map(to_dense_object(nested_object(std::forward<Arg>(arg))).data())};
        }
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg> or Eigen3::eigen_TriangularView<Arg>)
      {
        return OpenKalman::diagonal_of(nested_object(std::forward<Arg>(arg)));
      }
      else if constexpr (Eigen3::eigen_Identity<Arg>)
      {
        auto f = [](const auto& a, const auto& b) { return std::min(a, b); };
        auto dim = value::operation{f, get_index_dimension_of<0>(arg), get_index_dimension_of<1>(arg)};
        return make_constant<Arg, Scalar, 1>(dim);
      }
      else if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        decltype(auto) m = to_native_matrix(std::forward<Arg>(arg));
        using M = decltype(m);
        using M_noref = std::remove_reference_t<M>;
        using MatrixType = std::conditional_t<std::is_lvalue_reference_v<M>, M_noref, const M_noref>;
        return Eigen::Diagonal<MatrixType, 0> {std::forward<M>(m)};
      }
      else
      {
        return OpenKalman::diagonal_of(to_native_matrix(std::forward<Arg>(arg)));
      }
    }


    template<typename Arg, typename Factor0 = std::integral_constant<std::size_t, 1>, typename Factor1 = std::integral_constant<std::size_t, 1>>
    static auto
    broadcast(Arg&& arg, const Factor0& factor0 = Factor0{}, const Factor1& factor1 = Factor1{})
    {
      constexpr int F0 = []{
        if constexpr (value::fixed<Factor0>) return static_cast<std::size_t>(Factor0{});
        else return Eigen::Dynamic;
      }();
      constexpr int F1 = []{
        if constexpr (value::fixed<Factor1>) return static_cast<std::size_t>(Factor1{});
        else return Eigen::Dynamic;
      }();

      using IndexType = typename std::decay_t<Arg>::Index;

      decltype(auto) m = to_native_matrix(std::forward<Arg>(arg));
      using M = decltype(m);
      using R = Eigen::Replicate<std::remove_reference_t<M>, F0, F1>;
      if constexpr (value::fixed<Factor0> and value::fixed<Factor1>)
        return R {std::forward<M>(m)};
      else
        return R {std::forward<M>(m), static_cast<IndexType>(factor0), static_cast<IndexType>(factor1)};
    }

  private:

    // Only to be used in a non-evaluated context
    template<typename Op, typename...S>
    static constexpr auto dummy_op(Op op, S...s)
    {
      if constexpr (std::is_invocable_v<Op, S...>) return op(s...);
      else if constexpr (std::is_invocable_v<Op, std::size_t, std::size_t>) return op(std::size_t{0}, std::size_t{0});
      else return op(std::size_t{0});
    }


    template<typename Op, typename...Args>
    struct EigenNaryOp;

    template<typename Op, typename Arg>
    struct EigenNaryOp<Op, Arg> { using type = Eigen::CwiseUnaryOp<Op, Arg>; };

    template<typename Op, typename Arg1, typename Arg2>
    struct EigenNaryOp<Op, Arg1, Arg2> { using type = Eigen::CwiseBinaryOp<Op, Arg1, Arg2>; };

    template<typename Op, typename Arg1, typename Arg2, typename Arg3>
    struct EigenNaryOp<Op, Arg1, Arg2, Arg3> { using type = Eigen::CwiseTernaryOp<Op, Arg1, Arg2, Arg3>; };

  public:

#ifdef __cpp_concepts
    template<coordinate::pattern...Ds, typename Operation, indexible...Args> requires
      (sizeof...(Ds) <= 2) and (sizeof...(Args) <= 3) and
      (value::number<std::invoke_result_t<Operation, scalar_type_of_t<Args>...>> or
      (sizeof...(Args) == 0 and
        (value::number<std::invoke_result_t<Operation, std::conditional_t<true, std::size_t, Ds>...>> or
        value::number<std::invoke_result_t<Operation, std::size_t>>)))
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<sizeof...(Ds) <= 2 and sizeof...(Args) <= 3 and
      (coordinate::pattern<Ds> and ...) and (indexible<Args> and ...) and
      (value::number<typename std::invoke_result<Operation, typename scalar_type_of<Args>::type...>::type> or
        (sizeof...(Args) == 0 and
          (value::number<typename std::invoke_result<Operation, std::conditional_t<true, std::size_t, Ds>...>::type> or
          value::number<typename std::invoke_result<Operation, std::size_t>::type>))), int> = 0>
#endif
    static auto
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      decltype(auto) op = Eigen3::native_operation(std::forward<Operation>(operation));
      using Op = decltype(op);
      using Scalar = decltype(dummy_op(operation, std::declval<scalar_type_of_t<Args>>()...));

      if constexpr (sizeof...(Args) == 0)
      {
        using P = dense_writable_matrix_t<T, Layout::none, Scalar, std::tuple<Ds...>>;
        return Eigen::CwiseNullaryOp<std::remove_reference_t<Op>, P> {
          static_cast<typename P::Index>(get_size(std::get<0>(tup))),
          static_cast<typename P::Index>(get_size(std::get<1>(tup))),
          std::forward<Op>(op)};
      }
      else
      {
        auto seq = std::index_sequence_for<Ds...>{};
        using CW = typename EigenNaryOp<std::decay_t<Op>, std::remove_reference_t<Args>...>::type;
        return CW {std::forward<Args>(args)..., std::forward<Op>(op)};
      }
    }


#ifdef __cpp_concepts
    template<std::size_t...indices, typename BinaryFunction, typename Arg> requires ((indices <= 1) and ...)
#else
    template<std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<((indices <= 1) and ...), int> = 0>
#endif
    static auto
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        auto&& op = Eigen3::native_operation(std::forward<BinaryFunction>(b));
        using Op = decltype(op);

        if constexpr (((indices == 0) or ...) and ((indices == 1) or ...))
        {
          return std::forward<Arg>(arg).redux(std::forward<Op>(op)); // Complete reduction, which will be a scalar.
        }
        else
        {
          constexpr auto dir = ((indices == 0) and ...) ? Eigen::Vertical : Eigen::Horizontal;
          using ROp = Eigen::internal::member_redux<std::decay_t<Op>, scalar_type_of_t<Arg>>;
          decltype(auto) m = to_native_matrix(std::forward<Arg>(arg));
          using M = decltype(m);
          using M_noref = std::remove_reference_t<M>;
          using MatrixType = std::conditional_t<std::is_lvalue_reference_v<M>, M_noref, const M_noref>;
          return Eigen::PartialReduxExpr<MatrixType, ROp, dir> {std::forward<M>(m), ROp{std::forward<Op>(op)}};
        }
      }
      else
      {
        return reduce<indices...>(std::forward<BinaryFunction>(b), to_native_matrix(std::forward<Arg>(arg)));
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        using UnaryOp = Eigen::internal::scalar_conjugate_op<scalar_type_of_t<Arg>>;
        decltype(auto) m = to_native_matrix(std::forward<Arg>(arg));
        using M = decltype(m);
        return Eigen::CwiseUnaryOp<UnaryOp, std::remove_reference_t<M>> {std::forward<M>(m)};
      }
      else if constexpr (Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
      {
        return std::forward<Arg>(arg).conjugate();
      }
      else if constexpr (triangular_matrix<Arg>)
      {
        return OpenKalman::conjugate(TriangularAdapter {std::forward<Arg>(arg)});
      }
      else
      {
        return conjugate(to_native_matrix(std::forward<Arg>(arg)));
      }
      // Note: the global conjugate function already handles Eigen::DiagonalMatrix and Eigen::DiagonalWrapper
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_matrix_general<Arg>)
      {
        decltype(auto) m = to_native_matrix(std::forward<Arg>(arg));
        using M = decltype(m);
        using M_noref = std::remove_reference_t<M>;
        using MatrixType = std::conditional_t<std::is_lvalue_reference_v<M>, M_noref, const M_noref>;
        return Eigen::Transpose<MatrixType> {std::forward<M>(m)};
      }
      else if constexpr (Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
      {
        return std::forward<Arg>(arg).transpose();
      }
      else if constexpr (triangular_matrix<Arg>)
      {
        static_assert(triangle_type_of_v<Arg> == TriangleType::upper or triangle_type_of_v<Arg> == TriangleType::lower);
        constexpr auto t = triangular_matrix<Arg, TriangleType::upper> ? TriangleType::lower : TriangleType::upper;
        return OpenKalman::make_triangular_matrix<t>(transpose(to_native_matrix(std::forward<Arg>(arg))));
      }
      else
      {
        return transpose(to_native_matrix(std::forward<Arg>(arg)));
      }
      // Note: the global transpose function already handles zero, constant, diagonal, constant-diagonal, and symmetric cases.
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (not Eigen3::eigen_dense_general<Arg>)
#else
    template<typename Arg, std::enable_if_t<not Eigen3::eigen_dense_general<Arg>, int> = 0>
#endif
    static constexpr decltype(auto)
    adjoint(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
      {
        return std::forward<Arg>(arg).adjoint();
      }
      else if constexpr (triangular_matrix<Arg>)
      {
        static_assert(triangle_type_of_v<Arg> == TriangleType::upper or triangle_type_of_v<Arg> == TriangleType::lower);
        constexpr auto t = triangular_matrix<Arg, TriangleType::upper> ? TriangleType::lower : TriangleType::upper;
        return OpenKalman::make_triangular_matrix<t>(OpenKalman::adjoint(to_native_matrix(std::forward<Arg>(arg))));
      }
      else
      {
        return OpenKalman::adjoint(to_native_matrix(std::forward<Arg>(arg)));
      }
      // Note: the global adjoint function already handles zero, constant, diagonal, non-complex, and hermitian cases.
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_matrix_general<Arg, true>)
        return std::forward<Arg>(arg).determinant();
      else if constexpr (Eigen3::eigen_array_general<Arg, true>)
        return std::forward<Arg>(arg).matrix().determinant();
      else
        return to_native_matrix(std::forward<Arg>(arg)).determinant();
      // Note: the global determinant function already handles Eigen::TriangularView, Eigen::DiagonalMatrix, and Eigen::DiagonalWrapper
    }


#ifdef __cpp_concepts
    template<typename A, Eigen3::eigen_general B>
#else
    template<typename A, typename B, std::enable_if_t<Eigen3::eigen_general<B>, int> = 0>
#endif
    static constexpr auto
    sum(A&& a, B&& b)
    {
      constexpr HermitianAdapterType h = hermitian_adapter_type_of_v<A, B> == HermitianAdapterType::any ?
        HermitianAdapterType::lower : hermitian_adapter_type_of_v<A, B>;

      auto f = [](auto&& x) -> decltype(auto) {
        constexpr bool herm = hermitian_matrix<A> and hermitian_matrix<B>;
        if constexpr ((triangle_type_of_v<A, B> != TriangleType::any and triangular_adapter<decltype(x)>) or (herm and hermitian_adapter<decltype(x), h>))
          return nested_object(std::forward<decltype(x)>(x));
        else if constexpr (herm and hermitian_adapter<decltype(x)>)
          return OpenKalman::transpose(nested_object(std::forward<decltype(x)>(x)));
        else
          return std::forward<decltype(x)>(x);
      };

      auto op = Eigen::internal::scalar_sum_op<scalar_type_of_t<A>, scalar_type_of_t<B>>{};

      decltype(auto) a_wrap = to_native_matrix(f(std::forward<A>(a)));
      using AWrap = decltype(a_wrap);
      decltype(auto) b_wrap = to_native_matrix(f(std::forward<B>(b)));
      using BWrap = decltype(b_wrap);
      using CW = Eigen::CwiseBinaryOp<decltype(op), std::remove_reference_t<AWrap>, std::remove_reference_t<BWrap>>;
      CW s {std::forward<AWrap>(a_wrap), std::forward<BWrap>(b_wrap), std::move(op)};

      if constexpr (triangle_type_of_v<A, B> != TriangleType::any) return OpenKalman::make_triangular_matrix<triangle_type_of_v<A, B>>(std::move(s));
      else if constexpr (hermitian_matrix<A> and hermitian_matrix<B>) return make_hermitian_matrix<h>(std::move(s));
      else return s;
    }

  private:

    template<typename Arg, typename S, typename Op>
    static constexpr auto
    scalar_op_impl(Arg&& arg, S&& s, Op&& op)
    {
      using Scalar = scalar_type_of_t<Arg>;
      using ConstOp = Eigen::internal::scalar_constant_op<Scalar>;
      decltype(auto) m {to_native_matrix(std::forward<Arg>(arg))};
      using M = decltype(m);
      using PlainObjectType = dense_writable_matrix_t<M>;
      using Index = typename PlainObjectType::Index;
      auto c {Eigen::CwiseNullaryOp<ConstOp, PlainObjectType> {
        static_cast<Index>(get_index_dimension_of<0>(m)),
        static_cast<Index>(get_index_dimension_of<1>(m)),
        ConstOp {static_cast<Scalar>(value::to_number(s))}}};
      using CW = Eigen::CwiseBinaryOp<Op, std::remove_reference_t<M>, decltype(c)>;
      return CW {std::forward<M>(m), c, std::forward<Op>(op)};
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg, value::scalar S>
    static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
    template<typename Arg, typename S>
    static constexpr auto
#endif
    scalar_product(Arg&& arg, S&& s)
    {
      using Scalar = scalar_type_of_t<Arg>;
      using Op = Eigen::internal::scalar_product_op<Scalar, Scalar>;
      return scalar_op_impl(std::forward<Arg>(arg), std::forward<S>(s), Op{});
    }


#ifdef __cpp_concepts
    template<indexible Arg, value::scalar S>
    static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
    template<typename Arg, typename S>
    static constexpr auto
#endif
    scalar_quotient(Arg&& arg, S&& s)
    {
      using Scalar = scalar_type_of_t<Arg>;
      using Op = Eigen::internal::scalar_quotient_op<Scalar, Scalar>;
      return scalar_op_impl(std::forward<Arg>(arg), std::forward<S>(s), Op{});
    }


#ifdef __cpp_concepts
  template<Eigen3::eigen_general A, Eigen3::eigen_general B>
#else
  template<typename A, typename B, std::enable_if_t<(Eigen3::eigen_general<A> and Eigen3::eigen_general<B>), int> = 0>
#endif
    static constexpr auto
    contract(A&& a, B&& b)
    {
      if constexpr (diagonal_matrix<A> and not Eigen3::eigen_DiagonalWrapper<A> and not Eigen3::eigen_DiagonalMatrix<A>)
      {
        return contract(to_native_matrix(diagonal_of(std::forward<A>(a))).asDiagonal(), std::forward<B>(b));
      }
      else if constexpr (diagonal_matrix<B> and not Eigen3::eigen_DiagonalWrapper<B> and not Eigen3::eigen_DiagonalMatrix<B>)
      {
        return contract(std::forward<A>(a), to_native_matrix(diagonal_of(std::forward<B>(b))).asDiagonal());
      }
      else if constexpr (diagonal_matrix<A> or diagonal_matrix<B>)
      {
        decltype(auto) a_wrap = to_native_matrix(std::forward<A>(a));
        using AWrap = decltype(a_wrap);
        decltype(auto) b_wrap = to_native_matrix(std::forward<B>(b));
        using BWrap = decltype(b_wrap);
        using Prod = Eigen::Product<std::remove_reference_t<AWrap>, std::remove_reference_t<BWrap>, Eigen::LazyProduct>;
        return Prod {std::forward<AWrap>(a_wrap), std::forward<BWrap>(b_wrap)};
      }
      else if constexpr (not Eigen3::eigen_matrix_general<B>)
      {
        return contract(std::forward<A>(a), to_native_matrix(std::forward<B>(b)));
      }
      else if constexpr (Eigen3::eigen_matrix_general<A> or Eigen3::eigen_TriangularView<A> or Eigen3::eigen_SelfAdjointView<A>)
      {
        using Prod = Eigen::Product<std::remove_reference_t<A>, std::remove_reference_t<B>, Eigen::DefaultProduct>;
        return to_dense_object(Prod {std::forward<A>(a), std::forward<B>(b)});
      }
      else
      {
        return contract(to_native_matrix(std::forward<A>(a)), std::forward<B>(b));
      }
    }


#ifdef __cpp_concepts
    template<bool on_the_right, writable A, indexible B> requires Eigen3::eigen_dense_general<A> or
      Eigen3::eigen_DiagonalMatrix<A> or Eigen3::eigen_DiagonalWrapper<A>
#else
    template<bool on_the_right, typename A, typename B, std::enable_if_t<writable<A> and
      (Eigen3::eigen_dense_general<A> or Eigen3::eigen_DiagonalMatrix<A> or (Eigen3::eigen_DiagonalWrapper<A> and diagonal_adapter<A, 0>)), int> = 0>
#endif
    static A&
    contract_in_place(A& a, B&& b)
    {
      if constexpr (Eigen3::eigen_DiagonalMatrix<A> or Eigen3::eigen_DiagonalWrapper<A>)
      {
        static_assert(diagonal_matrix<B>);
        nested_object(a) = diagonal_of(a).array() * diagonal_of(std::forward<B>(b)).array();
      }
      else if constexpr (Eigen3::eigen_TriangularView<A>)
      {
        static_assert(triangular_matrix<B, triangle_type_of_v<A>>);
        nested_object(a) = contract(a, std::forward<B>(b));
      }
      else
      {
        auto&& ma = [](A& a) -> decltype(auto) {
          if constexpr (Eigen3::eigen_array_general<A>) return a.matrix();
          else return (a);
        }(a);

        if constexpr (on_the_right)
          return ma.applyOnTheRight(OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
        else
          return ma.applyOnTheLeft(OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
        return a;
      }
    }


#ifdef __cpp_concepts
    template<TriangleType triangle_type, Eigen3::eigen_SelfAdjointView A>
#else
    template<TriangleType triangle_type, typename A, std::enable_if_t<Eigen3::eigen_SelfAdjointView<A>, int> = 0>
#endif
    static constexpr auto
    cholesky_factor(A&& a)
    {
      using NestedMatrix = std::decay_t<nested_object_of_t<A>>;
      using Scalar = scalar_type_of_t<A>;
      auto dim = *is_square_shaped(a);

      if constexpr (constant_matrix<NestedMatrix>)
      {
        // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.

        auto s = constant_coefficient {a};

        if (value::to_number(s) < Scalar(0))
        {
          // Cholesky factor elements are complex, so throw an exception.
          throw (std::runtime_error("cholesky_factor of constant HermitianAdapter: result is indefinite"));
        }

        if constexpr(triangle_type == TriangleType::diagonal)
        {
          static_assert(diagonal_matrix<A>);
          return OpenKalman::to_diagonal(make_constant<A>(value::sqrt(s), dim, Dimensions<1>{}));
        }
        else if constexpr(triangle_type == TriangleType::lower)
        {
          auto euc_dim = get_size(dim);
          auto col0 = make_constant<A>(value::sqrt(s), euc_dim, Dimensions<1>{});
          auto othercols = make_zero<A>(euc_dim, euc_dim - Dimensions<1>{});
          return make_vector_space_adapter(OpenKalman::make_triangular_matrix<triangle_type>(concatenate_horizontal(col0, othercols)), dim, dim);
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto euc_dim = get_size(dim);
          auto row0 = make_constant<A>(value::sqrt(s), Dimensions<1>{}, dim);
          auto otherrows = make_zero<A>(euc_dim - Dimensions<1>{}, euc_dim);
          return make_vector_space_adapter(OpenKalman::make_triangular_matrix<triangle_type>(concatenate_vertical(row0, otherrows)), dim, dim);
        }
      }
      else
      {
        // For the general case, perform an LLT Cholesky decomposition.
        using M = dense_writable_matrix_t<A>;
        M b;
        auto LL_x = a.llt();
        if (LL_x.info() == Eigen::Success)
        {
          if constexpr((triangle_type == TriangleType::upper and hermitian_adapter_type_of_v<A> == HermitianAdapterType::upper) or
            triangle_type == TriangleType::lower and hermitian_adapter_type_of_v<A> == HermitianAdapterType::lower)
          {
            b = std::move(LL_x.matrixLLT());
          }
          else
          {
            constexpr unsigned int uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
            b.template triangularView<uplo>() = LL_x.matrixLLT().adjoint();
          }
        }
        else [[unlikely]]
        {
          // If covariance is not positive definite, use the more robust LDLT decomposition.
          auto LDL_x = nested_object(std::forward<A>(a)).ldlt();
          if ((not LDL_x.isPositive() and not LDL_x.isNegative()) or LDL_x.info() != Eigen::Success) [[unlikely]]
          {
            if (LDL_x.isPositive() and LDL_x.isNegative()) // Covariance is zero, even though decomposition failed.
            {
              if constexpr(triangle_type == TriangleType::lower)
                b.template triangularView<Eigen::Lower>() = make_zero(nested_object(a));
              else
                b.template triangularView<Eigen::Upper>() = make_zero(nested_object(a));
            }
            else // Covariance is indefinite, so throw an exception.
            {
              throw (std::runtime_error("cholesky_factor of HermitianAdapter: covariance is indefinite"));
            }
          }
          else if constexpr(triangle_type == TriangleType::lower)
          {
            b.template triangularView<Eigen::Lower>() =
              LDL_x.matrixL().toDenseMatrix() * LDL_x.vectorD().cwiseSqrt().asDiagonal();
          }
          else
          {
            b.template triangularView<Eigen::Upper>() =
              LDL_x.vectorD().cwiseSqrt().asDiagonal() * LDL_x.matrixU().toDenseMatrix();
          }
        }
        return TriangularAdapter<M, triangle_type> {std::move(b)};
      }
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
    {
      if constexpr (Eigen3::eigen_matrix_general<A>)
      {
        static_assert(writable<A>);
        constexpr auto s = significant_triangle == HermitianAdapterType::lower ? Eigen::Lower : Eigen::Upper;
        a.template selfadjointView<s>().template rankUpdate(std::forward<U>(u), alpha);
        return std::forward<A>(a);
      }
      else
      {
        return rank_update_hermitian<significant_triangle>(to_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      if constexpr (Eigen3::eigen_matrix_general<A>)
      {
        static_assert(writable<A>);
        constexpr auto t = triangle == TriangleType::lower ? Eigen::Lower : Eigen::Upper;
        using Scalar = scalar_type_of_t<A>;
        for (std::size_t i = 0; i < get_index_dimension_of<1>(u); i++)
        {
          if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(a, get_chip<1>(u, i), alpha) >= 0)
            throw (std::runtime_error("rank_update_triangular: product is not positive definite"));
        }
        return std::forward<A>(a);
      }
      else
      {
        return rank_update_triangular<triangle>(to_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


#ifdef __cpp_concepts
    template<bool must_be_unique, bool must_be_exact, typename A, typename B> requires Eigen3::eigen_matrix_general<B>
#else
    template<bool must_be_unique, bool must_be_exact, typename A, typename B, std::enable_if_t<Eigen3::eigen_matrix_general<B>, int> = 0>
#endif
    static constexpr auto
    solve(A&& a, B&& b)
    {
      using Scalar = scalar_type_of_t<A>;

      constexpr std::size_t a_rows = dynamic_dimension<A, 0> ? index_dimension_of_v<B, 0> : index_dimension_of_v<A, 0>;
      constexpr std::size_t a_cols = index_dimension_of_v<A, 1>;
      constexpr std::size_t b_cols = index_dimension_of_v<B, 1>;

      if constexpr (Eigen3::eigen_TriangularView<A>)
      {
        auto ret {Eigen::Solve {std::forward<A>(a), std::forward<B>(b)}};
        if constexpr (std::is_lvalue_reference_v<A> and std::is_lvalue_reference_v<B>) return ret;
        else return to_dense_object(std::move(ret));
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<A>)
      {
        constexpr auto uplo = hermitian_adapter_type_of_v<A> == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        auto v {std::forward<A>(a).template selfadjointView<uplo>()};
        auto llt {v.llt()};

        Eigen3::eigen_matrix_t<Scalar, a_cols, b_cols> ret;
        if (llt.info() == Eigen::Success)
        {
          ret = Eigen::Solve {llt, std::forward<B>(b)};
        }
        else [[unlikely]]
        {
          // A is semidefinite. Use LDLT decomposition instead.
          auto ldlt {v.ldlt()};
          if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
          {
            throw (std::runtime_error("Eigen solve (hermitian case): A is indefinite"));
          }
          ret = Eigen::Solve {ldlt, std::forward<B>(b)};
        }
        return ret;
      }
      else if constexpr (Eigen3::eigen_matrix_general<A>)
      {
        using Mat = Eigen3::eigen_matrix_t<Scalar, a_rows, a_cols>;
        if constexpr (must_be_exact or must_be_unique)
        {
          auto a_cols_rt = get_index_dimension_of<1>(a);
          auto qr {Eigen::ColPivHouseholderQR<Mat> {std::forward<A>(a)}};

          if constexpr (must_be_unique)
          {
            if (qr.rank() < a_cols_rt) throw std::runtime_error {"solve function requests a "
              "unique solution, but A is rank-deficient, so result X is not unique"};
          }

          auto res {to_dense_object(Eigen::Solve {std::move(qr), std::forward<B>(b)})};

          if constexpr (must_be_exact)
          {
            bool a_solution_exists = (a * res).isApprox(b, a_cols_rt * std::numeric_limits<scalar_type_of_t<A>>::epsilon());

            if (a_solution_exists) return res;
            else throw std::runtime_error {"solve function requests an exact solution, "
              "but the solution is only an approximation"};
          }
          else
          {
            return res;
          }
        }
        else
        {
          return to_dense_object(Eigen::Solve {Eigen::HouseholderQR<Mat> {std::forward<A>(a)}, std::forward<B>(b)});
        }
      }
      else
      {
        return solve<must_be_unique, must_be_exact>(to_native_matrix(std::forward<A>(a)), std::forward<B>(b));
      }
    }

  private:

      template<typename A>
      static constexpr auto
      QR_decomp_impl(A&& a)
      {
        using Scalar = scalar_type_of_t<A>;
        constexpr auto rows = index_dimension_of_v<A, 0>;
        constexpr auto cols = index_dimension_of_v<A, 1>;
        using MatrixType = Eigen3::eigen_matrix_t<Scalar, rows, cols>;
        using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;

        Eigen::HouseholderQR<MatrixType> QR {std::forward<A>(a)};

        if constexpr (dynamic_dimension<A, 1>)
        {
          auto rt_cols = get_index_dimension_of<1>(a);

          ResultType ret {rt_cols, rt_cols};

          if constexpr (dynamic_dimension<A, 0>)
          {
            auto rt_rows = get_index_dimension_of<0>(a);

            if (rt_rows < rt_cols)
              ret << QR.matrixQR().topRows(rt_rows),
                Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size>::Zero(rt_cols - rt_rows, rt_cols);
            else
              ret = QR.matrixQR().topRows(rt_cols);
          }
          else
          {
            if (rows < rt_cols)
              ret << QR.matrixQR().template topRows<rows>(),
                Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size>::Zero(rt_cols - rows, rt_cols);
            else
              ret = QR.matrixQR().topRows(rt_cols);
          }

          return ret;
        }
        else
        {
          ResultType ret;

          if constexpr (dynamic_dimension<A, 0>)
          {
            auto rt_rows = get_index_dimension_of<0>(a);

            if (rt_rows < cols)
              ret << QR.matrixQR().topRows(rt_rows),
              Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size>::Zero(cols - rt_rows, cols);
            else
              ret = QR.matrixQR().template topRows<cols>();
          }
          else
          {
            if constexpr (rows < cols)
              ret << QR.matrixQR().template topRows<rows>(), Eigen3::eigen_matrix_t<Scalar, cols - rows, cols>::Zero();
            else
              ret = QR.matrixQR().template topRows<cols>();
          }

          return ret;
        }
      }

  public:

    template<typename A>
    static constexpr auto
    LQ_decomposition(A&& a)
    {
      return OpenKalman::make_triangular_matrix<TriangleType::lower>(OpenKalman::adjoint(QR_decomp_impl(OpenKalman::adjoint(std::forward<A>(a)))));
    }


    template<typename A>
    static constexpr auto
    QR_decomposition(A&& a)
    {
      return OpenKalman::make_triangular_matrix<TriangleType::upper>(QR_decomp_impl(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_LIBRARY_INTERFACE_HPP
