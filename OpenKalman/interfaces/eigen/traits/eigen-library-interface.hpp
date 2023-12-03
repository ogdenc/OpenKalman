/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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
    using LibraryBase = Eigen3::EigenAdapterBase<Derived, T>;

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
          return static_cast<Scalar>(get_coeff(nested_matrix(std::forward<Arg>(arg)), i, 0));
        else
          return static_cast<Scalar>(0);
      }
      else if constexpr (Eigen3::eigen_TriangularView<Arg>)
      {
        constexpr int Mode {std::decay_t<Arg>::Mode};
        if ((i > j and (Mode & int{Eigen::Upper}) != 0x0) or (i < j and (Mode & int{Eigen::Lower}) != 0x0))
          return static_cast<Scalar>(0);
        else
          return static_cast<Scalar>(get_coeff(nested_matrix(std::forward<Arg>(arg)), i, j));
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        return get_coeff(nested_matrix(std::forward<Arg>(arg)), i, j);
      }
      else
      {
        constexpr bool use_coeffRef = (Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0 and
          std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>>;
        if constexpr (use_coeffRef)
          return arg.coeffRef(i, j);
        else if constexpr (std::is_lvalue_reference_v<Arg>)
          return arg.coeff(i, j);
        else
        {
          auto ret = std::forward<Arg>(arg).coeff(i, j);
          return ret;
        }
      }
    }


    template<typename Arg, typename Indices>
    static constexpr std::tuple<Eigen::Index, Eigen::Index>
    extract_indices(const Arg& arg, const Indices& indices)
    {
      auto sz = std::size(indices);
      if (sz == 2)
      {
        auto it = indices.begin();
        auto i = static_cast<std::size_t>(*it);
        return {i, static_cast<std::size_t>(*++it)};
      }
      else if (sz == 1)
      {
        return {static_cast<std::size_t>(*indices.begin()), 0};
      }
      else if (sz == 0)
      {
        return {0, 0};
      }
      else throw std::logic_error("Wrong number of indices on component access");
    }

  public:

#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<indexible Arg, std::ranges::input_range Indices> requires std::convertible_to<std::ranges::range_value_t<Indices>, const typename std::decay_t<Arg>::Index>
    static constexpr scalar_constant decltype(auto)
#else
    template<typename Arg, typename Indices>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      using Scalar = scalar_type_of_t<Arg>;
      auto [i, j] = extract_indices(arg, indices);
      if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        constexpr int Mode {std::decay_t<Arg>::Mode};
        bool transp = (i > j and (Mode & int{Eigen::Upper}) != 0) or (i < j and (Mode & int{Eigen::Lower}) != 0);
        if constexpr (complex_number<Scalar>)
        {
          using std::conj;
          if (transp) return static_cast<Scalar>(conj(get_coeff(std::as_const(arg), j, i)));
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


#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<indexible Arg, std::ranges::input_range Indices> requires (not std::is_const_v<Arg>) and
      std::convertible_to<std::ranges::range_value_t<Indices>, const typename Arg::Index> and
      std::assignable_from<decltype(get_coeff(std::declval<Arg&>(), 0, 0)), const scalar_type_of_t<Arg>&> and
      ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0) and
      (not diagonal_adapter<Arg>) and (not triangular_adapter<Arg>)
#else
    template<typename Arg, typename Indices, std::enable_if_t<(not std::is_const_v<Arg>) and
      std::is_assignable<decltype(get_coeff(std::declval<Arg&>(), 0, 0)), const scalar_type_of_t<Arg>&>::value and
      (Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0 and
      (not diagonal_adapter<Arg>) and (not triangular_adapter<Arg>), int> = 0>
#endif
    static void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      using Scalar = scalar_type_of_t<Arg>;
      auto [i, j] = extract_indices(arg, indices);
      if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        constexpr int Mode {std::decay_t<Arg>::Mode};
        bool transp = (i > j and (Mode & int{Eigen::Upper}) != 0) or (i < j and (Mode & int{Eigen::Lower}) != 0);
        if constexpr (complex_number<Scalar>)
        {
          using std::conj;
          if (transp) get_coeff(arg, j, i) = conj(s);
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


    template<typename Arg>
    static decltype(auto)
    to_native_matrix(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_matrix_general<Arg>)
      {
        return std::forward<Arg>(arg);
      }
      else if constexpr (Eigen3::eigen_array_general<Arg>)
      {
        return std::forward<Arg>(arg).matrix();
      }
      else if constexpr (directly_accessible<Arg>)
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
            if constexpr (static_index_value<S0, std::ptrdiff_t>) return static_cast<std::ptrdiff_t>(S0{});
            else return Eigen::Dynamic;
          }();
          constexpr int es1 = []() -> int {
            if constexpr (static_index_value<S1, std::ptrdiff_t>) return static_cast<std::ptrdiff_t>(S1{});
            else return Eigen::Dynamic;
          }();
          using IndexType = typename std::decay_t<Arg>::Index;
          auto is0 = static_cast<IndexType>(static_cast<std::ptrdiff_t>(s0));
          auto is1 = static_cast<IndexType>(static_cast<std::ptrdiff_t>(s1));

          if constexpr (static_index_value<S0, std::ptrdiff_t> and
            (es0 == 1 or (static_index_value<S1, std::ptrdiff_t> and es0 < es1)))
          {
            using M = Eigen::Matrix<Scalar, rows, cols, Eigen::ColMajor>;
            Eigen::Stride<es1, es0> strides {is1, is0};
            return Eigen::Map<M, Eigen::Unaligned, decltype(strides)> {strides};
          }
          else if constexpr (static_index_value<S1, std::ptrdiff_t> and
            (es1 == 1 or (static_index_value<S0, std::ptrdiff_t> and es1 < es0)))
          {
            using M = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;
            Eigen::Stride<es0, es1> strides {is0, is1};
            return Eigen::Map<M, Eigen::Unaligned, decltype(strides)> {strides};
          }
          else
          {
            if (is1 < is0)
            {
              using M = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;
              Eigen::Stride<es0, es1> strides {is0, is1};
              return Eigen::Map<M, Eigen::Unaligned, decltype(strides)> {strides};
            }
            else
            {
              using M = Eigen::Matrix<Scalar, rows, cols, Eigen::ColMajor>;
              Eigen::Stride<es1, es0> strides {is1, is0};
              return Eigen::Map<M, Eigen::Unaligned, decltype(strides)> {strides};
            }
          }
        }
        else
        {
          constexpr auto l = layout_of_v<Arg> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor;
          using M = Eigen::Matrix<Scalar, rows, cols, l>;
          return Eigen::Map<M> {internal::raw_data(arg)};
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

    template<std::size_t i, typename...D>
    static constexpr auto get_dim = dimension_size_of_v<std::tuple_element_t<i, std::tuple<D..., Dimensions<1>, Dimensions<1>>>>;

  public:

#ifdef __cpp_concepts
    template<Layout layout, scalar_type Scalar, vector_space_descriptor...D>
#else
    template<Layout layout, typename Scalar, typename...D, std::enable_if_t<scalar_type<Scalar> and
      (vector_space_descriptor<D> and ...), int> = 0>
#endif
    static auto make_default(D&&...d)
    {
      static_assert(layout != Layout::right or get_dim<0, D...> == 1 or get_dim<1, D...> != 1,
        "Eigen does not allow creation of a row-major column vector.");
      static_assert(layout != Layout::left or get_dim<0, D...> != 1 or get_dim<1, D...> == 1,
        "Eigen does not allow creation of a column-major row vector.");

      constexpr auto options =
        layout == Layout::right or (layout == Layout::none and get_dim<0, D...> == 1 and get_dim<1, D...> != 1) ?
        Eigen::RowMajor : Eigen::ColMajor;

      using IndexType = typename Eigen::Index;

      if constexpr (sizeof...(D) <= 2)
      {
        using M = writable_type<Scalar, get_dim<0, D...>, get_dim<1, D...>, options>;

        if constexpr (((dimension_size_of_v<D> == dynamic_size) or ...))
          return M(static_cast<IndexType>(get_dimension_size_of(d))...);
        else
          return M {};
      }
      else if constexpr ((dynamic_vector_space_descriptor<D> or ...))
      {
        return Eigen::Tensor<Scalar, sizeof...(D), options, IndexType> {static_cast<IndexType>(get_dimension_size_of(d))...};
      }
      else
      {
        return Eigen::TensorFixedSize<Scalar, Eigen::Sizes<std::decay_t<D>::value...>, options, IndexType> {};
      }
    }


#ifdef __cpp_concepts
    template<Layout layout, writable M, std::convertible_to<scalar_type_of_t<M>> Arg, std::convertible_to<scalar_type_of_t<M>> ... Args>
      requires (layout == Layout::right) or (layout == Layout::left)
#else
    template<Layout layout, typename M, typename Arg, typename...Args, std::enable_if_t<writable<M> and
      (layout == Layout::right or layout == Layout::left) and
      std::conjunction<std::is_convertible<Arg, typename scalar_type_of<M>::type>,
          std::is_convertible<Args, typename scalar_type_of<M>::type>...>::value, int> = 0>
#endif
    static M&& fill_with_elements(M&& m, const Arg arg, const Args ... args)
    {
      if constexpr (layout == Layout::left) ((m.transpose() << arg), ... , args);
      else ((m << arg), ... , args);
      return std::forward<M>(m);
    }


#ifdef __cpp_concepts
    template<scalar_constant<CompileTimeStatus::unknown> C, typename...Ds> requires (sizeof...(Ds) <= 2)
    static constexpr constant_matrix<CompileTimeStatus::unknown> auto
#else
    template<typename C, typename...Ds, std::enable_if_t<
      scalar_constant<C, CompileTimeStatus::unknown> and (sizeof...(Ds) <= 2), int> = 0>
    static constexpr auto
#endif
    make_constant_matrix(C&& c, Ds&&...ds)
    {
      auto value = get_scalar_constant_value(std::forward<C>(c));
      using Scalar = std::decay_t<decltype(value)>;
      using N = dense_writable_matrix_t<T, Layout::none, Scalar, std::decay_t<Ds>...>;
      if constexpr (sizeof...(Ds) == 2)
        return N::Constant(static_cast<typename N::Index>(get_dimension_size_of(std::forward<Ds>(ds)))..., value);
      else if constexpr (sizeof...(Ds) == 1)
        return N::Constant(static_cast<typename N::Index>(get_dimension_size_of(std::forward<Ds>(ds)))..., 1, value);
      else // sizeof...(Ds) == 0
        return N::Constant(value);
      // Note: We don't address orders greater than 2 because Eigen::Tensor's constant has substantial limitations.
    }


    template<typename Scalar, typename D>
    static constexpr auto
    make_identity_matrix(D&& d)
    {
      if constexpr (dimension_size_of_v<D> == dynamic_size)
      {
        return to_diagonal(make_constant_matrix_like<T, Scalar, 1>(std::forward<D>(d), Dimensions<1>{}));
      }
      else
      {
        constexpr auto n {static_cast<int>(dimension_size_of_v<D>)};
        return Eigen::Matrix<Scalar, n, n>::Identity();
      }
    }


#ifdef __cpp_concepts
    template<TriangleType t, Eigen3::eigen_dense_general Arg>
#else
    template<TriangleType t, typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg>, int> = 0>
#endif
    static constexpr auto
    make_triangular_matrix(Arg& arg)
    {
      constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
      return arg.template triangularView<TriMode>();
    }


#ifdef __cpp_concepts
    template<HermitianAdapterType t, Eigen3::eigen_dense_general Arg>
#else
    template<HermitianAdapterType t, typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg>, int> = 0>
#endif
    static constexpr auto
    make_hermitian_adapter(Arg& arg)
    {
      constexpr auto HMode = t == HermitianAdapterType::upper ? Eigen::Upper : Eigen::Lower;
      return arg.template selfadjointView<HMode>();
    }


    template<typename Arg, typename...Begin, typename...Size>
    static auto
    get_block(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size)
    {
      static_assert(0 < sizeof...(Begin) and sizeof...(Begin) <= 2);
      static_assert(sizeof...(Begin) == sizeof...(Size));

      // \todo If Eigen implements Block for non-dense matrices, this can be changed to Eigen3::eigen_general.
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        auto b0 = std::get<0>(begin);
        auto b1 = [](const auto& begin){
          if constexpr (sizeof...(Begin) >= 2) return std::get<1>(begin);
          else return std::integral_constant<std::size_t, 0>{};
        }(begin);

        auto s0 = std::get<0>(size);
        auto s1 = [](const auto& size){
          if constexpr (sizeof...(Size) >= 2) return std::get<1>(size);
          else return std::integral_constant<std::size_t, 1>{};
        }(size);

        constexpr int S0 = static_cast<int>(static_index_value<decltype(s0)> ? static_cast<std::size_t>(s0) : Eigen::Dynamic);
        constexpr int S1 = static_cast<int>(static_index_value<decltype(s1)> ? static_cast<std::size_t>(s1) : Eigen::Dynamic);
        using B = Eigen::Block<std::remove_reference_t<Arg>, S0, S1>;

        if constexpr ((static_index_value<Size> and ...)) return make_self_contained<Arg>(B(arg, b0, b1));
        else return make_self_contained<Arg>(B(arg, b0, b1, s0, s1));
      }
      else return make_self_contained(get_block(to_native_matrix(std::forward<Arg>(arg)), begin, size));
    }


    template<typename Arg, typename Block, typename...Begin>
    static constexpr Arg&
    set_block(Arg& arg, Block&& block, const Begin&...begin)
    {
      static_assert(0 < sizeof...(Begin) and sizeof...(Begin) <= 2);

      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        set_block(nested_matrix(arg), std::forward<Block>(block), begin...);
        return arg;
      }
      else
      {
        static_assert(Eigen3::eigen_dense_general<Arg>);

        auto [b0, b1] = [](Eigen::Index bs0, Eigen::Index bs1, const auto&...){ return std::tuple {bs0, bs1}; }
          (static_cast<std::size_t>(begin)..., 0_uz, 0_uz);

        if constexpr (Eigen3::eigen_block<Block>)
        {
          if (std::addressof(arg) == std::addressof(block.nestedExpression()) and b0 == block.startRow() and b1 == block.startCol())
            return arg;
        }

        constexpr auto Bx0 = static_cast<int>(index_dimension_of_v<Block, 0>);
        constexpr auto Bx1 = static_cast<int>(index_dimension_of_v<Block, 1>);
        using Bk = Eigen::Block<std::remove_reference_t<Arg>, Bx0, Bx1>;

        if constexpr (not has_dynamic_dimensions<Block>)
        {
          if constexpr (std::is_assignable_v<Bk&, Block&&>)
            Bk {arg, b0, b1} = std::forward<Block>(block);
          else
            Bk {arg, b0, b1} = to_native_matrix(std::forward<Block>(block));
        }
        else
        {
          auto s0 = static_cast<Eigen::Index>(get_index_dimension_of<0>(block));
          auto s1 = static_cast<Eigen::Index>(get_index_dimension_of<1>(block));

          if constexpr (std::is_assignable_v<Bk&, Block&&>)
            Bk {arg, b0, b1, s0, s1} = std::forward<Block>(block);
          else
            Bk {arg, b0, b1, s0, s1} = to_native_matrix(std::forward<Block>(block));
        }

        return arg;
      }
    }

  private:

#ifdef __cpp_concepts
    template<typename A>
#else
    template<typename A, typename = void>
#endif
    struct pass_through_eigenwrapper : std::false_type {};

#ifdef __cpp_concepts
    template<Eigen3::eigen_wrapper A>
    struct pass_through_eigenwrapper<A>
#else
    template<typename A>
    struct pass_through_eigenwrapper<A, std::enable_if_t<Eigen3::eigen_wrapper<A>>>
#endif
      : std::bool_constant<Eigen3::eigen_dense_general<nested_matrix_of_t<A>> or diagonal_adapter<nested_matrix_of_t<A>> or
        triangular_adapter<nested_matrix_of_t<A>> or hermitian_adapter<nested_matrix_of_t<A>>> {};

  public:

    template<TriangleType t, typename A, typename B>
    static decltype(auto)
    set_triangle(A&& a, B&& b)
    {
      if constexpr (Eigen3::eigen_MatrixWrapper<A> or Eigen3::eigen_ArrayWrapper<A> or pass_through_eigenwrapper<A>::value)
      {
        return internal::set_triangle<t>(nested_matrix(std::forward<A>(a)), std::forward<B>(b));
      }
      else if constexpr (not Eigen3::eigen_dense_general<A>)
      {
        return set_triangle<t>(to_native_matrix(std::forward<A>(a)), std::forward<B>(b));
      }
      else
      {
        decltype(auto) aw = make_dense_writable_matrix_from(std::forward<A>(a));

        auto awview = [](auto&& aw) {
          if constexpr (t == TriangleType::diagonal) return aw.diagonal();
          else return aw.template triangularView<t == TriangleType::upper ? Eigen::Upper : Eigen::Lower>();
        }(std::forward<decltype(aw)>(aw));

        if constexpr (t == TriangleType::diagonal)
        {
          auto&& bdiag = OpenKalman::diagonal_of(std::forward<B>(b));
          if constexpr (std::is_assignable_v<decltype((awview)), decltype(bdiag)>)
            awview = std::forward<decltype(bdiag)>(bdiag);
          else
            awview = to_native_matrix(std::forward<decltype(bdiag)>(bdiag));
        }
        else
        {
          if constexpr (std::is_assignable_v<decltype((awview)), B&&>)
            awview = std::forward<B>(b);
          else
            awview = to_native_matrix(std::forward<B>(b));
        }

        return aw;
      }
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_dense_general Arg> requires square_matrix<Arg> or dimension_size_of_index_is<Arg, 0, 1> or
      std::is_lvalue_reference_v<nested_matrix_of_t<Eigen::DiagonalWrapper<std::remove_reference_t<Arg>>>>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 0, 1> or
      std::is_lvalue_reference_v<typename nested_matrix_of<Eigen::DiagonalWrapper<std::remove_reference_t<Arg>>>::type>), int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      if constexpr (not vector<Arg>) if (not get_is_vector(arg)) throw std::invalid_argument {
        "Argument of to_diagonal must have 1 column; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};
      if constexpr (square_matrix<Arg> or dimension_size_of_index_is<Arg, 0, 1>)
        return internal::FixedSizeAdapter {std::forward<Arg>(arg)};
      else
        return Eigen::DiagonalWrapper<std::remove_reference_t<Arg>> {arg};
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_SelfAdjointView Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_SelfAdjointView<Arg>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, the SelfAdjointView wrapper doesn't matter, and otherwise, the following will throw an exception:
      return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_TriangularView Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_TriangularView<Arg>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, the TriangularView wrapper doesn't matter, and otherwise, the following will throw an exception:
      return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg)));
    }


    template<typename Arg>
#ifdef __cpp_concepts
    static constexpr vector decltype(auto)
#else
    static constexpr decltype(auto)
#endif
    diagonal_of(Arg&& arg)
    {
      auto d = get_is_square(arg);
      if (not d) throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead it has " +
          std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
          std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (Eigen3::eigen_DiagonalWrapper<Arg>)
      {
        decltype(auto) diag {nested_matrix(std::forward<Arg>(arg))}; //< must be nested_matrix(...) rather than .diagonal() because of const_cast
        using Diag = decltype(diag);
        using EigenTraits = Eigen::internal::traits<std::decay_t<Diag>>;
        constexpr auto rows = EigenTraits::RowsAtCompileTime;
        constexpr auto cols = EigenTraits::ColsAtCompileTime;

        static_assert(cols != 1, "For Eigen::DiagonalWrapper<T> interface, T should never be a column vector "
                                 "because diagonal_of function handles this case.");
        if constexpr (cols == 0)
        {
          return std::forward<Diag>(diag);
        }
        else if constexpr (rows == 1 or rows == 0)
        {
          return OpenKalman::transpose(std::forward<Diag>(diag));
        }
        else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
        {
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data(),
            get_index_dimension_of<0>(diag) * get_index_dimension_of<1>(diag))};
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data())};
        }
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg> or Eigen3::eigen_TriangularView<Arg>)
      {
        // Assume there are no dangling references
        return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (Eigen3::eigen_Identity<Arg>)
      {
        constexpr std::size_t dim = dynamic_dimension<Arg, 0> ? index_dimension_of_v<Arg, 1> : index_dimension_of_v<Arg, 0>;
        if constexpr (dim == dynamic_size) return make_constant_matrix_like<Arg, Scalar, 1>(*d);
        else return make_constant_matrix_like<Arg, Scalar, 1>(Dimensions<dim>{});
      }
      else if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        auto&& n = nested_matrix(std::forward<Arg>(arg));
        using N = decltype(n);
        if constexpr (Eigen3::eigen_general<N, true>)
        {
          return OpenKalman::diagonal_of(std::forward<N>(n));
        }
        else if constexpr (Eigen3::eigen_array_general<N>)
        {
          auto ret {make_self_contained<N>(std::forward<N>(n).matrix().diagonal())};
          return ret;
        }
        else
        {
          auto ret {make_self_contained<N>(std::forward<N>(n).diagonal())};
          return ret;
        }
      }
      else if constexpr (Eigen3::eigen_array_general<Arg>)
      {
        auto ret {make_self_contained<Arg>(std::forward<Arg>(arg).matrix().diagonal())};
        return ret;
      }
      else
      {
        auto ret {make_self_contained<Arg>(std::forward<Arg>(arg).diagonal())};
        return ret;
      }
    }


    template<typename Arg, typename Factor0, typename Factor1 = std::integral_constant<std::size_t, 1>>
    static auto
    broadcast(Arg&& arg, const Factor0& factor0, const Factor1& factor1 = Factor1{})
    {
      constexpr int F0 = []{
        if constexpr (static_index_value<Factor0>) return static_cast<std::size_t>(Factor0{});
        else return Eigen::Dynamic;
      }();
      constexpr int F1 = []{
        if constexpr (static_index_value<Factor1>) return static_cast<std::size_t>(Factor1{});
        else return Eigen::Dynamic;
      }();

      using R = Eigen::Replicate<std::decay_t<Arg>, F0, F1>;
      using IndexType = typename std::decay_t<Arg>::Index;

      if constexpr (static_index_value<Factor0> and static_index_value<Factor1>)
        return R {std::forward<Arg>(arg)};
      else
        return R {std::forward<Arg>(arg), static_cast<IndexType>(factor0), static_cast<IndexType>(factor1)};
    }

  private:

    template<typename Arg, typename...Ds, std::size_t...Is>
    static constexpr bool is_replicatable(std::index_sequence<Is...>)
    {
      return (... and (
        dynamic_dimension<Arg, Is> or dynamic_vector_space_descriptor<std::tuple_element_t<Is, std::tuple<Ds...>>> or
        dimension_size_of_index_is<Arg, Is, 1> or
        dimension_size_of_index_is<Arg, Is, dimension_size_of_v<std::tuple_element_t<Is, std::tuple<Ds...>>>>));
    }


    template<typename...Ds, typename Arg, std::size_t...Ix_Ds>
    static decltype(auto)
    replicate_arg_impl(const std::tuple<Ds...>& d_tup, Arg&& arg, std::index_sequence<Ix_Ds...>)
    {
      constexpr auto factors = []{
        if constexpr (sizeof...(Ds) == 0)
          return std::tuple {1, 1};
        else if constexpr (sizeof...(Ds) == 1)
          return std::tuple {(dimension_size_of_v<Ds> == dynamic_size or dynamic_dimension<Arg, Ix_Ds> ?
            Eigen::Dynamic : static_cast<int>(dimension_size_of_v<Ds> / index_dimension_of_v<Arg, Ix_Ds>))..., 1};
        else
          return std::tuple {(dimension_size_of_v<Ds> == dynamic_size or dynamic_dimension<Arg, Ix_Ds> ?
            Eigen::Dynamic : static_cast<int>(dimension_size_of_v<Ds> / index_dimension_of_v<Arg, Ix_Ds>))...};
      }();

      using R = Eigen::Replicate<std::decay_t<Arg>, std::get<0>(factors), std::get<1>(factors)>;

      if constexpr (not (dynamic_vector_space_descriptor<Ds> or ...) and not (dynamic_dimension<Arg, Ix_Ds> or ...))
      {
        if constexpr ((dimension_size_of_index_is<Arg, Ix_Ds, dimension_size_of_v<Ds>> and ...))
          return std::forward<Arg>(arg);
        else
          return R {std::forward<Arg>(arg)};
      }
      else
      {
        auto [f0, f1] = [](const auto& d_tup, const auto& arg){
          if constexpr (sizeof...(Ds) == 0)
            return std::tuple {1, 1};
          else if constexpr (sizeof...(Ds) == 1)
            return std::tuple {static_cast<int>(get_dimension_size_of(std::get<Ix_Ds>(d_tup)) / get_index_dimension_of<Ix_Ds>(arg))..., 1};
          else
            return std::tuple {static_cast<int>(get_dimension_size_of(std::get<Ix_Ds>(d_tup)) / get_index_dimension_of<Ix_Ds>(arg))...};
        }(d_tup, arg);

        return R {std::forward<Arg>(arg), f0, f1};
      }
    }


    /**
     * \internal
     * \brief Replicate an object, if necessary, to expand any 1D indices to fill a particular shape.
     * \details If Arg already has the shape defined by <code>sizeof...(Ds)</ref> return the argument unchanged.
     * \param d_tup A tuple of \ref vector_space_descriptor (of type Ds) defining the resulting tensor.
     * Any trailing omitted descriptors will be considered as \ref Dimension<1>.
     * \tparam Arg The argument to be replicated.
     */
#ifdef __cpp_concepts
    template<typename...Ds, typename Arg> requires (sizeof...(Ds) <= 2) and
      (is_replicatable<Arg, Ds...>(std::make_index_sequence<sizeof...(Ds)>{}))
#else
    template<typename...Ds, typename Arg, std::enable_if_t<(sizeof...(Ds) <= 2) and
      (is_replicatable<Arg, Ds...>(std::make_index_sequence<sizeof...(Ds)>{}))>>
#endif
    static decltype(auto)
    replicate(const std::tuple<Ds...>& d_tup, Arg&& arg)
    {
      return replicate_arg_impl(d_tup, std::forward<Arg>(arg), std::index_sequence_for<Ds...> {});
    }


    // Only to be used in a non-evaluated context
    template<typename Op, typename...S>
    static constexpr auto dummy_op(Op op, S...s)
    {
      if constexpr (std::is_invocable_v<Op, S...>) return op(s...);
      else if constexpr (std::is_invocable_v<Op, std::size_t, std::size_t>) return op(std::size_t{0}, std::size_t{0});
      else return op(std::size_t{0});
    }


    template<typename...Ds, typename Operation, typename...Args>
    static auto
    n_ary_operation_impl(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      auto&& op = Eigen3::native_operation(std::forward<Operation>(operation));
      using Op = decltype(op);
      using Scalar = decltype(dummy_op(operation, std::declval<scalar_type_of_t<Args>>()...));

      if constexpr (sizeof...(Args) == 0)
      {
        using P = dense_writable_matrix_t<T, Layout::none, Scalar, Ds...>;
        return Eigen::CwiseNullaryOp<std::decay_t<Op>, P> {
          static_cast<typename P::Index>(get_dimension_size_of(std::get<0>(tup))),
          static_cast<typename P::Index>(get_dimension_size_of(std::get<1>(tup))),
          std::forward<Op>(op)};
      }
      else if constexpr (sizeof...(Args) == 1)
      {
        return make_self_contained<Args...>(Eigen::CwiseUnaryOp {
          replicate(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
      else if constexpr (sizeof...(Args) == 2)
      {
        return make_self_contained<Args...>(Eigen::CwiseBinaryOp<std::decay_t<Op>,
          std::decay_t<decltype(replicate(tup, std::forward<Args>(args)))>...> {
          replicate(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
      else
      {
        return make_self_contained<Args...>(Eigen::CwiseTernaryOp<std::decay_t<Op>,
          std::decay_t<decltype(replicate(tup, std::forward<Args>(args)))>...> {
          replicate(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
    }

  public:

#ifdef __cpp_concepts
    template<vector_space_descriptor...Ds, typename Operation, indexible...Args> requires
      (sizeof...(Ds) <= 2) and (sizeof...(Args) <= 3) and
      (scalar_type<std::invoke_result_t<Operation, scalar_type_of_t<Args>...>> or
      (sizeof...(Args) == 0 and
        (scalar_type<std::invoke_result_t<Operation, std::conditional_t<true, std::size_t, Ds>...>> or
        scalar_type<std::invoke_result_t<Operation, std::size_t>>)))
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<sizeof...(Ds) <= 2 and sizeof...(Args) <= 3 and
      (vector_space_descriptor<Ds> and ...) and (indexible<Args> and ...) and
      (scalar_type<typename std::invoke_result<Operation, typename scalar_type_of<Args>::type...>::type> or
        (sizeof...(Args) == 0 and
          (scalar_type<typename std::invoke_result<Operation, std::conditional_t<true, std::size_t, Ds>...>::type> or
          scalar_type<typename std::invoke_result<Operation, std::size_t>::type>))), int> = 0>
#endif
    static auto
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      auto ret {n_ary_operation_impl(tup, std::forward<Operation>(operation), std::forward<Args>(args)...)};
      if constexpr ((euclidean_vector_space_descriptor<Ds> and ...) and (all_fixed_indices_are_euclidean<Args> and ...))
        return ret;
      else
        return make_matrix<Ds...>(std::move(ret));
    }


#ifdef __cpp_concepts
    template<std::size_t...indices, typename BinaryFunction, typename Arg> requires (sizeof...(indices) <= 2)
#else
    template<std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<(sizeof...(indices) <= 2), int> = 0>
#endif
    static auto
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        auto&& op = Eigen3::native_operation(std::forward<BinaryFunction>(b));
        using Op = decltype(op);

        if constexpr (sizeof...(indices) == index_count_v<Arg>)
          return std::forward<Arg>(arg).redux(std::forward<Op>(op)); // Complete reduction, which will be a scalar.
        else if constexpr (((indices == 0) and ...))
          return make_self_contained<Arg>(std::forward<Arg>(arg).colwise().redux(std::forward<Op>(op)));
        else
          return make_self_contained<Arg>(std::forward<Arg>(arg).rowwise().redux(std::forward<Op>(op)));
      }
      else
      {
        return reduce<indices...>(std::forward<BinaryFunction>(b), to_native_matrix(std::forward<Arg>(arg)));
      }
    }

    // to_euclidean not defined--rely on default

    // from_euclidean not defined--rely on default

    // wrap_angles not defined--rely on default


    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg) noexcept
    {
      // The global conjugate function already handles DiagonalMatrix and DiagonalWrapper
      return std::forward<Arg>(arg).conjugate();
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::eigen_general<nested_matrix_of_t<Arg>, true>)
          return transpose(nested_matrix(std::forward<Arg>(arg)));
        else
          return std::forward<Arg>(arg).transpose(); // Rely on inherited Eigen transpose method
      }
      else if constexpr (Eigen3::eigen_matrix_general<Arg, true> or Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
        return std::forward<Arg>(arg).transpose();
      else if constexpr (Eigen3::eigen_array_general<Arg, true>)
        return std::forward<Arg>(arg).matrix().transpose();
      else if constexpr (triangular_matrix<Arg>)
        return OpenKalman::transpose(TriangularMatrix {std::forward<Arg>(arg)});
      else
        return to_native_matrix(std::forward<Arg>(arg)).transpose();
      // Note: the global transpose function already handles zero, constant, constant-diagonal, and symmetric cases.
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::eigen_general<nested_matrix_of_t<Arg>, true>)
          return adjoint(nested_matrix(std::forward<Arg>(arg)));
        else
          return std::forward<Arg>(arg).adjoint(); // Rely on inherited Eigen adjoint method
      }
      else if constexpr (Eigen3::eigen_matrix_general<Arg, true> or Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
        return std::forward<Arg>(arg).adjoint();
      else if constexpr (Eigen3::eigen_array_general<Arg, true>)
        return std::forward<Arg>(arg).matrix().adjoint();
      else if constexpr (triangular_matrix<Arg>)
        return OpenKalman::adjoint(TriangularMatrix {std::forward<Arg>(arg)});
      else
        return to_native_matrix(std::forward<Arg>(arg)).adjoint();
      // Note: the global adjoint function already handles zero, constant, diagonal, non-complex, and hermitian cases.
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_matrix_general<Arg, true>)
        return std::forward<Arg>(arg).determinant();
      else if constexpr (Eigen3::eigen_array_general<Arg, true>)
        return std::forward<Arg>(arg).matrix().determinant();
      else
        return to_native_matrix(std::forward<Arg>(arg)).determinant();
      // Note: the global determinant function already handles TriangularView, DiagonalMatrix, and DiagonalWrapper
    }


    template<typename A, typename B>
    static constexpr auto
    sum(A&& a, B&& b)
    {
      if constexpr (Eigen3::eigen_general<B>)
      {
        return make_self_contained<A, B>(std::forward<A>(a) + std::forward<B>(b));
      }
      else
      {
        return sum(std::forward<A>(a), to_native_matrix(std::forward<B>(b)));
      }
    }


    template<typename A, typename B>
    static constexpr auto
    contract(A&& a, B&& b)
    {
      if constexpr (diagonal_adapter<A>)
      {
        if constexpr (Eigen3::eigen_DiagonalWrapper<A> or Eigen3::eigen_DiagonalMatrix<A>)
          return make_self_contained<A, B>(std::forward<A>(a) * OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
        else
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<A>(a))).asDiagonal() *
            OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (diagonal_adapter<B>)
      {
        if constexpr (Eigen3::eigen_DiagonalWrapper<B> or Eigen3::eigen_DiagonalMatrix<B>)
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) * std::forward<B>(b));
        else
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) *
            OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<B>(b))).asDiagonal());
      }
      else if constexpr (triangular_adapter<A>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<A>(a))).template triangularView<uplo>() *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (triangular_adapter<B>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        auto prod = OpenKalman::to_native_matrix<T>(std::forward<A>(a));
        prod.applyOnTheRight(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<B>(b))).template triangularView<uplo>());
        return prod;
      }
      else if constexpr (hermitian_adapter<A>)
      {
        constexpr auto uplo = hermitian_adapter<A, HermitianAdapterType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<A>(a))).template selfadjointView<uplo>() *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (hermitian_adapter<B>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        auto prod = OpenKalman::to_native_matrix<T>(std::forward<A>(a));
        prod.applyOnTheRight(OpenKalman::to_native_matrix<T>(nested_matrix(std::forward<B>(b))).template selfadjointView<uplo>());
        return prod;
      }
      else
      {
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
    }


#ifdef __cpp_concepts
    template<bool on_the_right, writable A, indexible B> requires Eigen3::eigen_dense_general<A>
#else
    template<bool on_the_right, typename A, typename B, std::enable_if_t<writable<A> and Eigen3::eigen_dense_general<A>,int> = 0>
#endif
    static A&
    contract_in_place(A& a, B&& b)
    {
      auto&& ma = [](A& a) -> decltype(auto) {
        if constexpr (Eigen3::eigen_array_general<A, true>) return a.matrix();
        else return (a);
      }(a);

      if constexpr (on_the_right)
        return ma.applyOnTheRight(OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      else
        return ma.applyOnTheLeft(OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      return a;
    }


    template<TriangleType triangle_type, typename A>
    static constexpr auto
    cholesky_factor(A&& a) noexcept
    {
      using NestedMatrix = std::decay_t<nested_matrix_of_t<A>>;
      using Scalar = scalar_type_of_t<A>;
      constexpr auto dim = index_dimension_of_v<A, 0>;
      using M = dense_writable_matrix_t<A>;

      if constexpr (std::is_same_v<
        const NestedMatrix, const typename Eigen::MatrixBase<NestedMatrix>::ConstantReturnType>)
      {
        // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.

        auto s = nested_matrix(std::forward<A>(a)).functor()();

        if (s < Scalar(0))
        {
          // Cholesky factor elements are complex, so throw an exception.
          throw (std::runtime_error("Cholesky_factor of constant SelfAdjointMatrix: covariance is indefinite"));
        }

        if constexpr(triangle_type == TriangleType::diagonal)
        {
          static_assert(diagonal_matrix<A>);
          auto vec = make_constant_matrix_like<A>(square_root(s), Dimensions<dim>{}, Dimensions<1>{});
          return DiagonalMatrix<decltype(vec)> {vec};
        }
        else if constexpr(triangle_type == TriangleType::lower)
        {
          auto col0 = make_constant_matrix_like<A>(square_root(s), Dimensions<dim>{}, Dimensions<1>{});
          auto othercols = make_zero_matrix_like<A>(get_vector_space_descriptor<0>(a), get_vector_space_descriptor<0>(a) - 1);
          return TriangularMatrix<M, triangle_type> {concatenate_horizontal(col0, othercols)};
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto row0 = make_constant_matrix_like<A>(square_root(s), Dimensions<1>{}, Dimensions<dim>{});
          auto otherrows = make_zero_matrix_like<A>(get_vector_space_descriptor<0>(a) - 1, get_vector_space_descriptor<0>(a));
          return TriangularMatrix<M, triangle_type> {concatenate_vertical(row0, otherrows)};
        }
      }
      else
      {
        // For the general case, perform an LLT Cholesky decomposition.
        M b;
        auto LL_x = a.view().llt();
        if (LL_x.info() == Eigen::Success)
        {
          if constexpr(triangle_type == hermitian_adapter_type_of_v<A>)
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
          auto LDL_x = nested_matrix(std::forward<A>(a)).ldlt();
          if ((not LDL_x.isPositive() and not LDL_x.isNegative()) or LDL_x.info() != Eigen::Success) [[unlikely]]
          {
            if (LDL_x.isPositive() and LDL_x.isNegative()) // Covariance is zero, even though decomposition failed.
            {
              if constexpr(triangle_type == TriangleType::lower)
                b.template triangularView<Eigen::Lower>() = make_zero_matrix_like(nested_matrix(a));
              else
                b.template triangularView<Eigen::Upper>() = make_zero_matrix_like(nested_matrix(a));
            }
            else // Covariance is indefinite, so throw an exception.
            {
              throw (std::runtime_error("Cholesky_factor of SelfAdjointMatrix: covariance is indefinite"));
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
        return TriangularMatrix<M, triangle_type> {std::move(b)};
      }
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      if constexpr (OpenKalman::Eigen3::eigen_ArrayWrapper<A>)
      {
        return rank_update_self_adjoint<significant_triangle>(nested_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
      else if constexpr (OpenKalman::Eigen3::eigen_array_general<A, true>)
      {
        return rank_update_self_adjoint<significant_triangle>(std::forward<A>(a).matrix(), std::forward<U>(u), alpha);
      }
      else
      {
        static_assert(writable<A>);
        constexpr auto s = significant_triangle == HermitianAdapterType::lower ? Eigen::Lower : Eigen::Upper;
        a.template selfadjointView<s>().template rankUpdate(std::forward<U>(u), alpha);
        return std::forward<A>(a);
      }
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      if constexpr (OpenKalman::Eigen3::eigen_ArrayWrapper<A>)
      {
        return rank_update_triangular<triangle>(nested_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
      else if constexpr (OpenKalman::Eigen3::eigen_array_general<A, true>)
      {
        return rank_update_triangular<triangle>(std::forward<A>(a).matrix(), std::forward<U>(u), alpha);
      }
      else
      {
        static_assert(writable<A>);
        constexpr auto t = triangle == TriangleType::lower ? Eigen::Lower : Eigen::Upper;
        using Scalar = scalar_type_of_t<A>;
        for (std::size_t i = 0; i < get_index_dimension_of<1>(u); i++)
        {
          if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(a, get_column(u, i), alpha) >= 0)
            throw (std::runtime_error("rank_update_triangular: product is not positive definite"));
        }
        return std::forward<A>(a);
      }
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr auto
    solve(A&& a, B&& b)
    {
      using Scalar = scalar_type_of_t<A>;

      constexpr std::size_t a_rows = dynamic_dimension<A, 0> ? index_dimension_of_v<B, 0> : index_dimension_of_v<A, 0>;
      constexpr std::size_t a_cols = index_dimension_of_v<A, 1>;
      constexpr std::size_t b_cols = index_dimension_of_v<B, 1>;

      if constexpr (not Eigen3::eigen_matrix_general<A, true>)
      {
        auto&& n = OpenKalman::to_native_matrix(std::forward<A>(a));
        static_assert(Eigen3::eigen_matrix_general<decltype(n), true>);
        return solve<must_be_unique, must_be_exact>(std::forward<decltype(n)>(n), std::forward<B>(b));
      }
      else if constexpr (triangular_matrix<A>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(
          Eigen::Solve {std::forward<A>(a).template triangularView<uplo>(), std::forward<B>(b)});
      }
      else if constexpr (hermitian_matrix<A>)
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
      else
      {
        if constexpr (must_be_exact or must_be_unique or true)
        {
          auto a_cols_rt = get_index_dimension_of<1>(a);
          Eigen::ColPivHouseholderQR<Eigen3::eigen_matrix_t<Scalar, a_rows, a_cols>> QR {std::forward<A>(a)};
          if constexpr (must_be_unique)
          {
            if (QR.rank() < a_cols_rt) throw std::runtime_error {"solve function requests a "
              "unique solution, but A is rank-deficient, so result X is not unique"};
          }

          auto res = QR.solve(std::forward<B>(b));

          if constexpr (must_be_exact)
          {
            bool a_solution_exists = (a*res).isApprox(b, a_cols_rt * std::numeric_limits<scalar_type_of_t<A>>::epsilon());

            if (a_solution_exists)
              return make_self_contained(std::move(res));
            else
              throw std::runtime_error {"solve function requests an exact solution, "
              "but the solution is only an approximation"};
          }
          else
          {
            return make_self_contained(std::move(res));
          }
        }
        else
        {
          Eigen::HouseholderQR<Eigen3::eigen_matrix_t<Scalar, a_rows, a_cols>> QR {std::forward<A>(a)};
          return make_self_contained(QR.solve(std::forward<B>(b)));
        }
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
      return make_triangular_matrix<TriangleType::lower>(make_self_contained(adjoint(QR_decomp_impl(adjoint(std::forward<A>(a))))));
    }


    template<typename A>
    static constexpr auto
    QR_decomposition(A&& a)
    {
      return make_triangular_matrix<TriangleType::upper>(QR_decomp_impl(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_LIBRARY_INTERFACE_HPP
