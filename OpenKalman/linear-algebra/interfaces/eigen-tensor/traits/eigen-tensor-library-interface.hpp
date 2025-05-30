/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Library interface for native Eigen Tensor types.
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_LIBRARY_INTERFACE_HPP
#define OPENKALMAN_EIGEN_TENSOR_LIBRARY_INTERFACE_HPP

#include <type_traits>
#include <tuple>
#include <random>
#include "linear-algebra/interfaces/eigen-tensor/details/eigen-tensor-forward-declarations.hpp"


// ------------------- //
//  library_interface  //
// ------------------- //

namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<Eigen3::eigen_tensor_general<true> T>
  struct library_interface<T>
#else
  template<typename T>
  struct library_interface<T, std::enable_if_t<Eigen3::eigen_tensor_general<T, true>>>
#endif
  {
  private:

    using IndexType = typename Eigen::internal::traits<T>::Index;

  public:

    template<typename Derived>
    using LibraryBase = Eigen3::EigenTensorAdapterBase<Derived, T>;


#ifdef __cpp_lib_concepts
    template<indexible Arg, std::size_t NumIndices> requires (NumIndices == index_count_v<Arg>)
    static constexpr values::scalar decltype(auto)
#else
    template <typename Arg, std::size_t NumIndices, std::enable_if_t<NumIndices == index_count_v<Arg>, int> = 0>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const std::array<IndexType, NumIndices>& indices)
    {
      if constexpr ((Eigen::internal::traits<T>::Flags & Eigen::LvalueBit) != 0)
        return Eigen::TensorEvaluator<std::remove_reference_t<Arg>, Eigen::DefaultDevice> {std::forward<Arg>(arg), Eigen::DefaultDevice{}}.coeffRef(indices);
      else
        return Eigen::TensorEvaluator<std::remove_reference_t<Arg>, Eigen::DefaultDevice> {std::forward<Arg>(arg), Eigen::DefaultDevice{}}.coeff(indices);
    }

  private:

    template<std::size_t ix_count, typename I, typename End, typename...Ixs>
    static constexpr decltype(auto)
    make_ix_array(I i, const End& end, Ixs...indices)
    {
      if constexpr (ix_count == 0)
      {
        if (i != end) throw std::logic_error("Too many indices on component access");
        return std::array<IndexType, sizeof...(Ixs)> {indices...};
      }
      else
      {
        if (i == end) throw std::logic_error("Not enough indices on component access");
        auto this_ix {*i};
        return make_ix_array<ix_count - 1>(++i, end, indices..., static_cast<IndexType>(this_ix));
      }
    }

  public:

#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires
      std::convertible_to<std::ranges::range_value_t<Indices>, const typename std::decay_t<Arg>::Index>
    static constexpr values::scalar decltype(auto)
#else
    template<typename Arg, typename Indices>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      constexpr std::size_t ix_count = Eigen::internal::traits<T>::NumDimensions;
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      return get_component(std::forward<Arg>(arg), make_ix_array<ix_count>(ranges::begin(indices), ranges::end(indices)));
    }


#ifdef __cpp_lib_concepts
    template<indexible Arg, std::size_t NumIndices> requires (NumIndices == index_count_v<Arg>) and (not std::is_const_v<Arg>)
#else
    template <typename Arg, std::size_t NumIndices, std::enable_if_t<NumIndices == index_count<Arg>::value and
      not std::is_const<Arg>::value, int> = 0>
#endif
    static void
    set_component(Arg& arg, const scalar_type_of_t<T>& s, const std::array<IndexType, NumIndices>& indices)
    {
      Eigen::TensorEvaluator<Arg, Eigen::DefaultDevice> {arg, Eigen::DefaultDevice{}}.coeffRef(indices) = s;
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires
      std::convertible_to<std::ranges::range_value_t<Indices>, const typename std::decay_t<Arg>::Index> and
      ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0) and (not std::is_const_v<Arg>)
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0)  and (not std::is_const_v<Arg>), int> = 0>
#endif
    static void
    set_component(Arg& arg, const scalar_type_of_t<T>& s, const Indices& indices)
    {
      constexpr std::size_t ix_count = Eigen::internal::traits<T>::NumDimensions;
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      set_component(arg, s, make_ix_array<ix_count>(ranges::begin(indices), ranges::end(indices)));
    }

  private:

    template<auto l, typename Arg, std::size_t...Is>
    static constexpr auto
    make_TensorMap_impl(const Arg& arg, std::index_sequence<Is...>)
    {
      using M = Eigen::TensorFixedSize<scalar_type_of_t<Arg>, Eigen::Sizes<static_cast<std::ptrdiff_t>(index_dimension_of_v<Arg, Is>)...>, l, IndexType>;
      return Eigen::TensorMap<const M, l> {internal::raw_data(arg), index_dimension_of_v<Arg, Is>...};
    }


    template<auto l, typename Arg>
    static constexpr auto
    make_TensorMap(const Arg& arg)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
      {
        using M = Eigen::Tensor<scalar_type_of_t<Arg>, index_count_v<Arg>, l, IndexType>;
        return Eigen::TensorMap<const M, l>{internal::raw_data(arg)};
      }
      else
      {
        return make_TensorMap_impl<l>(arg, std::make_index_sequence<index_count_v<Arg>>{});
      }
    }

  public:

    template<typename Arg>
    static decltype(auto)
    to_native_matrix(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_tensor_wrapper<Arg>)
      {
        return std::forward<Arg>(arg);
      }
      else if constexpr (internal::library_wrapper<Arg>)
      {
        return to_native_matrix(OpenKalman::nested_object(std::forward<Arg>(arg)));
      }
      else if constexpr (not Eigen3::eigen_tensor_general<Arg> and directly_accessible<Arg> and std::is_lvalue_reference_v<Arg>)
      {
        if constexpr (layout_of_v<Arg> == Layout::stride and internal::has_static_strides<Arg>)
        {
          auto strides = internal::strides(arg);
          constexpr std::ptrdiff_t stride0 = std::get<0>(strides);
          constexpr std::ptrdiff_t strideN = std::get<index_count_v<Arg> - 1>(strides);
          constexpr auto l = stride0 > strideN ? Eigen::RowMajor : Eigen::ColMajor;
          return std::apply(
            [](auto&& map, auto&&...s){
              return Eigen::TensorStridingOp {map, std::array<std::size_t, index_count_v<Arg>>{s...}};
            },
            std::tuple_cat(make_TensorMap<l>(arg), std::move(strides)));
        }
        else
        {
          constexpr auto l = layout_of_v<Arg> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor;
          return make_TensorMap<l>(arg);
        }
      }
      else
      {
        return Eigen3::make_eigen_tensor_wrapper(std::forward<Arg>(arg));
      }
    }


#ifdef __cpp_concepts
    template<typename To, Eigen3::eigen_tensor_general From> requires (std::assignable_from<To&, From&&>)
#else
    template<typename To, typename From, std::enable_if_t<Eigen3::eigen_tensor_general<From> and std::is_assignable_v<To&, From&&>, int> = 0>
#endif
    static void assign(To& a, From&& b)
    {
      a = std::forward<From>(b);
    }


    template<Layout layout, typename Scalar, typename...D>
    static auto make_default(D&&...d)
    {
      constexpr auto options = layout == Layout::right ? Eigen::RowMajor : Eigen::ColMajor;
      if constexpr (((dynamic_pattern<D>) or ...))
        return Eigen::Tensor<Scalar, sizeof...(D), options, IndexType>(static_cast<IndexType>(get_dimension(d))...);
      else
        return Eigen::TensorFixedSize<Scalar, Eigen::Sizes<static_cast<std::ptrdiff_t>(coordinates::dimension_of_v<D>)...>, options, IndexType> {};
    }


#ifdef __cpp_concepts
    template<Layout layout, writable Arg, std::convertible_to<scalar_type_of_t<Arg>> ... Scalars>
      requires (layout == Layout::right) or (layout == Layout::left)
#else
    template<Layout layout, typename Arg, typename...Scalars, std::enable_if_t<writable<Arg> and
      (layout == Layout::right or or layout == Layout::left) and
      std::conjunction<std::is_convertible<Scalars, typename scalar_type_of<Arg>::type>::value...>::value, int> = 0>
#endif
    static void fill_components(Arg& arg, const Scalars ... scalars)
    {
      if constexpr (layout == Layout::left)
        arg.swap_layout().setValues({scalars...});
      else
        arg.setValues({scalars...});
    }


#ifdef __cpp_concepts
    template<values::dynamic C, typename...Ds> requires (... and (not dynamic_pattern<Ds>))
    static constexpr constant_matrix auto
#else
    template<typename C, typename...Ds, std::enable_if_t<values::dynamic<C> and
      (... and (not dynamic_pattern<Ds>)), int> = 0>
    static constexpr auto
#endif
    make_constant(C&& c, Ds&&...ds)
    {
      auto value = values::to_number(std::forward<C>(c));
      using Scalar = std::decay_t<decltype(value)>;
      auto m = make_default<Layout::none, Scalar>(std::forward<Ds>(ds)...);
      // m will be a dangling reference to TensorFixedSize, but Eigen only references its static dimensions, so there is no bug
      return Eigen::TensorCwiseNullaryOp {m, Eigen::internal::scalar_constant_op<Scalar>(value)};
    }


    // make_identity_matrix not defined


    template<typename Arg, typename...Begin, typename...Size>
    static auto
    get_slice(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      auto offsets = std::apply([](auto&&...a) {
        return std::array<std::size_t, sizeof...(Begin)> {std::forward<decltype(a)>(a)...};
      }, begin);
      auto extents = std::apply([](auto&&...a) {
        return std::array<std::size_t, sizeof...(Size)> {std::forward<decltype(a)>(a)...};
      }, size);
      return to_native_matrix(std::forward<Arg>(arg)).slice(offsets, extents);
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_tensor_general<true> Arg, Eigen3::eigen_tensor_general Block, typename...Begin>
#else
    template<typename Arg, typename Block, typename...Begin, std::enable_if_t<
      Eigen3::eigen_tensor_general<Arg, true> and Eigen3::eigen_tensor_general<Block>, int> = 0>
#endif
    static constexpr void
    set_slice(Arg& arg, Block&& block, Begin...begin)
    {
      auto offsets = std::array {static_cast<std::size_t>(begin)...};
      auto extents = std::apply([](auto&&...a) {
        return std::array {get_dimension(std::forward<decltype(a)>(a))...}; }, all_vector_space_descriptors(block));
      arg.slice(offsets, extents) = std::forward<Block>(block);
    }

    /*
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
      : std::bool_constant<Eigen3::eigen_dense_general<nested_object_of_t<A>> or diagonal_adapter<nested_object_of_t<A>> or
        triangular_adapter<nested_object_of_t<A>> or hermitian_adapter<nested_object_of_t<A>>> {};

  public:

    template<TriangleType t, typename A, typename B>
    static decltype(auto) set_triangle(A&& a, B&& b)
    {
      if constexpr (Eigen3::eigen_MatrixWrapper<A> or Eigen3::eigen_ArrayWrapper<A> or pass_through_eigenwrapper<A>::value)
      {
        return internal::set_triangle<t>(nested_object(std::forward<A>(a)), std::forward<B>(b));
      }
      else if constexpr (not Eigen3::eigen_dense_general<A>)
      {
        return set_triangle<t>(Eigen3::make_eigen_wrapper(std::forward<A>(a)), std::forward<B>(b));
      }
      else
      {
        if constexpr (t == TriangleType::diagonal)
        {
          if constexpr (writable<A> and std::is_lvalue_reference_v<A>)
          {
            a.diagonal() = OpenKalman::diagonal_of(std::forward<B>(b));
            return std::forward<A>(a);
          }
          else
          {
            auto aw = to_dense_object(std::forward<A>(a));
            aw.diagonal() = OpenKalman::diagonal_of(std::forward<B>(b));
            return aw;
          }
        }
        else
        {
          decltype(auto) aw = [](A&& a) -> decltype(auto) {
            if constexpr (writable<A> and std::is_lvalue_reference_v<A>) return std::forward<A>(a);
            else return to_dense_object(std::forward<A>(a));
          }(std::forward<A>(a));

          auto tview = aw.template triangularView<t == TriangleType::upper ? Eigen::Upper : Eigen::Lower>();
          if constexpr (std::is_assignable_v<decltype(tview), B&&>)
            tview = std::forward<B>(b);
          else
            tview = Eigen3::make_eigen_wrapper(std::forward<B>(b));
          return std::forward<decltype(aw)>(aw);
        }
      }
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_dense_general Arg> requires
      std::is_lvalue_reference_v<nested_object_of_t<Eigen::DiagonalWrapper<std::remove_reference_t<Arg>>>>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg> and
      std::is_lvalue_reference_v<typename nested_object_of<Eigen::DiagonalWrapper<std::remove_reference_t<Arg>>>::type>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg& arg)
    {
      if constexpr (not vector<Arg>) if (not is_vector(arg)) throw std::invalid_argument {
        "Argument of to_diagonal must have 1 column; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};
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
      return OpenKalman::to_diagonal(nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<Eigen3::eigen_TriangularView Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_TriangularView<Arg>, int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg)
    {
      // If it is a column vector, the TriangularView wrapper doesn't matter, and otherwise, the following will thow an exception:
      return OpenKalman::to_diagonal(nested_object(std::forward<Arg>(arg)));
    }

  private:

    template<typename Arg, typename Id>
    static constexpr decltype(auto)
    diagonal_of_impl(Arg&& arg, Id&& id)
    {
      auto diag {[](Arg&& arg){
        if constexpr (Eigen3::eigen_array_general<Arg, true>)
          return make_self_contained<Arg>(std::forward<Arg>(arg).matrix().diagonal());
        else // eigen_matrix_general<Arg, true>
          return make_self_contained<Arg>(std::forward<Arg>(arg).diagonal());
      }(std::forward<Arg>(arg))};
      using Diag = const std::decay_t<decltype(diag)>;
      static_assert(vector<Diag>);

      using D = std::conditional_t<dynamic_dimension<Diag, 0> and fixed_pattern<Id>, std::decay_t<Id>, vector_space_descriptor_of_t<Diag, 0>>;

      if constexpr (not dynamic_dimension<Diag, 0> or dynamic_pattern<D>) return diag;
      else return internal::FixedSizeAdapter<Diag, D, Dimensions<1>> {std::move(diag)};
    }

  public:

#ifdef __cpp_concepts
    template<Eigen3::eigen_general<true> Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_general<Arg, true>, int> = 0>
#endif
    static constexpr decltype(auto)
    diagonal_of(Arg&& arg)
    {
      auto d = is_square_shaped(arg);
      if (not d) throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead it has " +
          std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
          std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (Eigen3::eigen_DiagonalWrapper<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        decltype(auto) diag {nested_object(std::forward<Arg>(arg))}; //< must be nested_object(...) rather than .diagonal() because of const_cast
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
          auto d {to_dense_object(std::forward<Diag>(diag))};
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          return M {M::Map(to_dense_object(std::forward<Diag>(diag)).data(),
            get_index_dimension_of<0>(diag) * get_index_dimension_of<1>(diag))};
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          return M {M::Map(to_dense_object(std::forward<Diag>(diag)).data())};
        }
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg> or Eigen3::eigen_TriangularView<Arg>)
      {
        // Assume there are no dangling references
        return OpenKalman::diagonal_of(nested_object(std::forward<Arg>(arg)));
      }
      else if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::eigen_general<nested_object_of_t<Arg>, true>)
          return OpenKalman::diagonal_of(nested_object(std::forward<Arg>(arg)));
        else
          return diagonal_of_impl(std::forward<Arg>(arg), *d);
      }
      else if constexpr (Eigen3::eigen_Identity<Arg>)
      {
        constexpr std::size_t dim = dynamic_dimension<Arg, 0> ? index_dimension_of_v<Arg, 1> : index_dimension_of_v<Arg, 0>;
        if constexpr (dim == dynamic_size) return make_constant<Arg, Scalar, 1>(*d, Dimensions<1>{});
        else return make_constant<Arg, Scalar, 1>(Dimensions<dim>{}, Dimensions<1>{});
      }
      else
      {
        return diagonal_of_impl(std::forward<Arg>(arg), *d);
      };
    }

  private:

    template<typename...Ds, typename...ArgDs, typename Arg, std::size_t...I>
    static decltype(auto)
    replicate_arg_impl(const std::tuple<Ds...>& p_tup, const std::tuple<ArgDs...>& arg_tup, Arg&& arg, std::index_sequence<I...>)
    {
      using R = Eigen::Replicate<std::decay_t<Arg>,
        (coordinates::dimension_of_v<Ds> == dynamic_size or coordinates::dimension_of_v<ArgDs> == dynamic_size ?
        Eigen::Dynamic : static_cast<IndexType>(coordinates::dimension_of_v<Ds> / coordinates::dimension_of_v<ArgDs>))...>;

      if constexpr (((coordinates::dimension_of_v<Ds> != dynamic_size) and ...) and
        ((coordinates::dimension_of_v<ArgDs> != dynamic_size) and ...))
      {
        if constexpr (((coordinates::dimension_of_v<Ds> == coordinates::dimension_of_v<ArgDs>) and ...))
          return std::forward<Arg>(arg);
        else
          return R {std::forward<Arg>(arg)};
      }
      else
      {
        auto ret = R {std::forward<Arg>(arg), static_cast<IndexType>(
          get_dimension(std::get<I>(p_tup)) / get_dimension(std::get<I>(arg_tup)))...};
        return ret;
      }
    }


    template<typename...Ds, typename Arg>
    static decltype(auto)
    replicate_arg(const std::tuple<Ds...>& p_tup, Arg&& arg)
    {
      return replicate_arg_impl(p_tup, all_vector_space_descriptors(arg), std::forward<Arg>(arg), std::index_sequence_for<Ds...> {});
    }

  public:

#ifdef __cpp_concepts
    template<typename...Ds, typename Operation, typename...Args> requires (sizeof...(Args) <= 3) and
      std::invocable<Operation&&, scalar_type_of_t<Args>...> and
      values::number<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>...>>
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<(sizeof...(Args) <= 3) and
      std::is_invocable<Operation&&, typename scalar_type_of<Args>::type...>::value and
      values::number<typename std::invoke_result<Operation&&, typename scalar_type_of<Args>::type...>::type>, int> = 0>
#endif
    static auto
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      auto&& op = nat_op(std::forward<Operation>(operation));
      using Op = decltype(op);

      if constexpr (sizeof...(Args) == 0)
      {
        using P = dense_writable_matrix_t<T, Layout::none, scalar_type_of_t<T>, std::tuple<Ds...>>;
        IndexType r = get_dimension(std::get<0>(tup));
        IndexType c = get_dimension(std::get<1>(tup));
        return Eigen::CwiseNullaryOp<std::decay_t<Op>, P> {r, c, std::forward<Op>(op)};
      }
      else if constexpr (sizeof...(Args) == 1)
      {
        return make_self_contained<Args...>(Eigen::CwiseUnaryOp {
          replicate_arg(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
      else if constexpr (sizeof...(Args) == 2)
      {
        return make_self_contained<Args...>(Eigen::CwiseBinaryOp<std::decay_t<Op>,
          std::decay_t<decltype(replicate_arg(tup, std::forward<Args>(args)))>...> {
          replicate_arg(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
      else
      {
        return make_self_contained<Args...>(Eigen::CwiseTernaryOp<std::decay_t<Op>,
          std::decay_t<decltype(replicate_arg(tup, std::forward<Args>(args)))>...> {
          replicate_arg(tup, std::forward<Args>(args))..., std::forward<Op>(op)});
      }
    }
    */


    /*
    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr auto
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        auto&& op = nat_op(std::forward<BinaryFunction>(b));
        using Op = decltype(op);

        if constexpr (sizeof...(indices) == 2) // reduce in both directions
        {
          return std::forward<Arg>(arg).redux(std::forward<Op>(op));
        }
        else
        {
          using OpWrapper = Eigen::internal::member_redux<std::decay_t<Op>, scalar_type_of_t<Arg>>;
          constexpr auto dir = ((indices == 0) and ...) ? Eigen::Vertical : Eigen::Horizontal;
          using P = Eigen::PartialReduxExpr<std::decay_t<Arg>, OpWrapper, dir>;
          return make_self_contained<Arg>(P {std::forward<Arg>(arg), OpWrapper {std::forward<Op>(op)}});
        }
      }
      else
      {
        return reduce<indices...>(std::forward<BinaryFunction>(b), Eigen3::make_eigen_wrapper(std::forward<Arg>(arg)));
      }
    }

    // to_euclidean not defined--rely on default

    // from_euclidean not defined--rely on default

    // wrap_angles not defined--rely on default


    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg)
    {
      // The global conjugate function already handles Eigen::DiagonalMatrix and Eigen::DiagonalWrapper
      return std::forward<Arg>(arg).conjugate();
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::eigen_general<nested_object_of_t<Arg>, true>)
          return transpose(nested_object(std::forward<Arg>(arg)));
        else
          return std::forward<Arg>(arg).transpose(); // Rely on inherited Eigen transpose method
      }
      else if constexpr (Eigen3::eigen_matrix_general<Arg, true> or Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
        return std::forward<Arg>(arg).transpose();
      else if constexpr (Eigen3::eigen_array_general<Arg, true>)
        return std::forward<Arg>(arg).matrix().transpose();
      else if constexpr (triangular_matrix<Arg>)
        return OpenKalman::transpose(TriangularAdapter {std::forward<Arg>(arg)});
      else
        return Eigen3::make_eigen_wrapper(std::forward<Arg>(arg)).transpose();
      // Note: the global transpose function already handles zero, constant, constant-diagonal, and symmetric cases.
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg)
    {
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::eigen_general<nested_object_of_t<Arg>, true>)
          return adjoint(nested_object(std::forward<Arg>(arg)));
        else
          return std::forward<Arg>(arg).adjoint(); // Rely on inherited Eigen adjoint method
      }
      else if constexpr (Eigen3::eigen_matrix_general<Arg, true> or Eigen3::eigen_TriangularView<Arg> or Eigen3::eigen_SelfAdjointView<Arg>)
        return std::forward<Arg>(arg).adjoint();
      else if constexpr (Eigen3::eigen_array_general<Arg, true>)
        return std::forward<Arg>(arg).matrix().adjoint();
      else if constexpr (triangular_matrix<Arg>)
        return OpenKalman::adjoint(TriangularAdapter {std::forward<Arg>(arg)});
      else
        return Eigen3::make_eigen_wrapper(std::forward<Arg>(arg)).adjoint();
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
        return Eigen3::make_eigen_wrapper(std::forward<Arg>(arg)).determinant();
      // Note: the global determinant function already handles Eigen::TriangularView, Eigen::DiagonalMatrix, and Eigen::DiagonalWrapper
    }


    template<typename A, typename B>
    static constexpr auto
    sum(A&& a, B&& b)
    {
      auto s = make_self_contained<A, B>(std::forward<A>(a) + std::forward<B>(b));
      if constexpr ((dynamic_dimension<decltype(s), 0> and (not dynamic_dimension<A, 0> or not dynamic_dimension<B, 0>)) or
          (dynamic_dimension<decltype(s), 1> and (not dynamic_dimension<A, 1> or not dynamic_dimension<B, 1>)))
      {
        using S0 = vector_space_descriptor_of_t<decltype(s), 0>; using S1 = vector_space_descriptor_of_t<decltype(s), 1>;
        using A0 = vector_space_descriptor_of_t<A, 0>; using A1 = vector_space_descriptor_of_t<A, 1>;
        using B0 = vector_space_descriptor_of_t<B, 0>; using B1 = vector_space_descriptor_of_t<B, 1>;

        using R = std::conditional_t<dynamic_pattern<S0>,
          std::conditional_t<dynamic_pattern<A0>, std::conditional_t<dynamic_pattern<B0>, S0, B0>, A0>, S0>;
        using C = std::conditional_t<dynamic_pattern<S1>,
          std::conditional_t<dynamic_pattern<A1>, std::conditional_t<dynamic_pattern<B1>, S1, B1>, A1>, S1>;
        return internal::FixedSizeAdapter<std::decay_t<decltype(s)>, R, C> {std::move(s)};
      }
      else return s;
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
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_object(std::forward<A>(a))).asDiagonal() *
            OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (diagonal_adapter<B>)
      {
        if constexpr (Eigen3::eigen_DiagonalWrapper<B> or Eigen3::eigen_DiagonalMatrix<B>)
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) * std::forward<B>(b));
        else
          return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(std::forward<A>(a)) *
            OpenKalman::to_native_matrix<T>(nested_object(std::forward<B>(b))).asDiagonal());
      }
      else if constexpr (triangular_adapter<A>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_object(std::forward<A>(a))).template triangularView<uplo>() *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (triangular_adapter<B>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        auto prod = OpenKalman::to_native_matrix<T>(std::forward<A>(a));
        prod.applyOnTheRight(OpenKalman::to_native_matrix<T>(nested_object(std::forward<B>(b))).template triangularView<uplo>());
        return prod;
      }
      else if constexpr (hermitian_adapter<A>)
      {
        constexpr auto uplo = hermitian_adapter<A, HermitianAdapterType::upper> ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(OpenKalman::to_native_matrix<T>(nested_object(std::forward<A>(a))).template selfadjointView<uplo>() *
          OpenKalman::to_native_matrix<T>(std::forward<B>(b)));
      }
      else if constexpr (hermitian_adapter<B>)
      {
        constexpr auto uplo = triangular_matrix<A, TriangleType::upper> ? Eigen::Upper : Eigen::Lower;
        auto prod = OpenKalman::to_native_matrix<T>(std::forward<A>(a));
        prod.applyOnTheRight(OpenKalman::to_native_matrix<T>(nested_object(std::forward<B>(b))).template selfadjointView<uplo>());
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
      auto&& ma = [](A& a){
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
    cholesky_factor(A&& a)
    {
      using NestedMatrix = std::decay_t<nested_object_of_t<A>>;
      using Scalar = scalar_type_of_t<A>;
      constexpr auto dim = index_dimension_of_v<A, 0>;
      using M = dense_writable_matrix_t<A>;

      if constexpr (std::is_same_v<
        const NestedMatrix, const typename Eigen::MatrixBase<NestedMatrix>::ConstantReturnType>)
      {
        // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.

        auto s = nested_object(std::forward<A>(a)).functor()();

        if (s < Scalar(0))
        {
          // Cholesky factor elements are complex, so throw an exception.
          throw (std::runtime_error("cholesky_factor of constant HermitianAdapter: covariance is indefinite"));
        }

        if constexpr(triangle_type == TriangleType::diagonal)
        {
          static_assert(diagonal_matrix<A>);
          auto vec = make_constant<A>(square_root(s), Dimensions<dim>{}, Dimensions<1>{});
          return DiagonalAdapter<decltype(vec)> {vec};
        }
        else if constexpr(triangle_type == TriangleType::lower)
        {
          auto col0 = make_constant<A>(square_root(s), Dimensions<dim>{}, Dimensions<1>{});
          auto othercols = make_zero<A>(get_vector_space_descriptor<0>(a), get_vector_space_descriptor<0>(a) - 1);
          return TriangularAdapter<M, triangle_type> {concatenate_horizontal(col0, othercols)};
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto row0 = make_constant<A>(square_root(s), Dimensions<1>{}, Dimensions<dim>{});
          auto otherrows = make_zero<A>(get_vector_space_descriptor<0>(a) - 1, get_vector_space_descriptor<0>(a));
          return TriangularAdapter<M, triangle_type> {concatenate_vertical(row0, otherrows)};
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
      if constexpr (OpenKalman::Eigen3::eigen_ArrayWrapper<A>)
      {
        return rank_update_hermitian<significant_triangle>(nested_object(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
      else if constexpr (OpenKalman::Eigen3::eigen_array_general<A, true>)
      {
        return rank_update_hermitian<significant_triangle>(std::forward<A>(a).matrix(), std::forward<U>(u), alpha);
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
        return rank_update_triangular<triangle>(nested_object(std::forward<A>(a)), std::forward<U>(u), alpha);
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
          if (Eigen::internal::llt_inplace<Scalar, t>::rankUpdate(a, get_chip<1>(u, i), alpha) >= 0)
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
    }*/

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TENSOR_LIBRARY_INTERFACE_HPP
