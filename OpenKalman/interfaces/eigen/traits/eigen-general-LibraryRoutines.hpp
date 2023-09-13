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

#ifndef OPENKALMAN_EIGEN_GENERAL_LIBRARYROUTINES_HPP
#define OPENKALMAN_EIGEN_GENERAL_LIBRARYROUTINES_HPP

#include <type_traits>
#include <tuple>
#include <random>


namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<Eigen3::eigen_general<true> T>
  struct LibraryRoutines<T>
#else
  template<typename T>
  struct LibraryRoutines<T, std::enable_if_t<Eigen3::eigen_general<T, true>>>
#endif
  {
    template<typename Derived>
    using LibraryBase = Eigen3::EigenAdapterBase<Derived, T>;


    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg)
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
        using index_type = index_type_of_t<Arg>;
        constexpr index_type rows = dynamic_dimension<Arg, 0> ? Eigen::Dynamic : index_dimension_of_v<Arg, 0>;
        constexpr index_type cols = dynamic_dimension<Arg, 1> ? Eigen::Dynamic : index_dimension_of_v<Arg, 1>;

        if constexpr (layout_of_v<Arg> == Layout::stride)
        {
          auto [s0, s1] = internal::strides(arg);
          using S0 = decltype(s0);
          using S1 = decltype(s1);
          constexpr auto es0 = static_index_value<S0> ? static_index_value_of_v<S0> : Eigen::Dynamic;
          constexpr auto es1 = static_index_value<S1> ? static_index_value_of_v<S1> : Eigen::Dynamic;
          auto is0 = static_cast<index_type>(s0);
          auto is1 = static_cast<index_type>(s1);

          if constexpr (static_index_value<S0> and (static_index_value_of_v<S0> == 1 or (static_index_value<S1> and es0 < es1)))
          {
            using M = Eigen::Matrix<Scalar, rows, cols, Eigen::ColMajor>;
            Eigen::Stride<es1, es0> strides {is1, is0};
            return Eigen::Map<M, Eigen::Unaligned, decltype(strides)> {strides};
          }
          else if constexpr (static_index_value<S1> and (static_index_value_of_v<S1> == 1 or (static_index_value<S0> and es1 < es0)))
          {
            using M = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;
            Eigen::Stride<es0, es1> strides {is0, is1};
            return Eigen::Map<M, Eigen::Unaligned, decltype(strides)> {strides};
          }
          else
          {
            if (static_cast<index_type>(s1) < static_cast<index_type>(s0))
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
        return Eigen3::EigenWrapper {std::forward<Arg>(arg)};
      }
    }

  private:

    using index_type = index_type_of_t<T>;

    template<typename Scalar, index_type...Is>
    using dense_type = std::conditional_t<Eigen3::eigen_array_general<T, true>,
      Eigen::Array<Scalar, Is...>, Eigen::Matrix<Scalar, Is...>>;

    template<typename Scalar, std::size_t...Is>
    using writable_type = dense_type<Scalar, (Is == dynamic_size ? Eigen::Dynamic : static_cast<index_type>(Is))...>;

  public:

    template<typename Scalar, typename...D>
    static auto make_default(D&&...d)
    {
      if constexpr (sizeof...(D) <= 2)
      {
        using M = writable_type<Scalar, dimension_size_of_v<D>...>;

        if constexpr (((dimension_size_of_v<D> == dynamic_size) or ...))
          return M(static_cast<index_type>(get_dimension_size_of(d))...);
        else
          return M {};
      }
      else if constexpr ((dynamic_index_descriptor<D> or ...))
      {
        using Tens = Eigen::Tensor<Scalar, sizeof...(D)>;
        return Tens {static_cast<typename Tens::Index>(get_dimension_size_of(d))...};
      }
      else
      {
        return Eigen::TensorFixedSize<Scalar, Eigen::Sizes<std::decay_t<D>::value...>> {};
      }
    }


#ifdef __cpp_concepts
    template<scalar_type Scalar, index_descriptor Drow, index_descriptor Dcol, std::convertible_to<Scalar> Arg, std::convertible_to<Scalar> ... Args>
#else
    template<typename Scalar, typename Drow, typename Dcol, typename Arg, typename...Args, std::enable_if_t<
      scalar_type<Scalar> and index_descriptor<Drow> and index_descriptor<Dcol> and
      std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make_from_elements(const std::tuple<Drow, Dcol>& d_tup, const Arg arg, const Args ... args)
    {
      constexpr std::size_t rows = dimension_size_of_v<Drow>;
      constexpr std::size_t columns = dimension_size_of_v<Dcol>;
      if constexpr (rows != dynamic_size and columns != dynamic_size)
        return ((Eigen3::eigen_matrix_t<Scalar, rows, columns> {} << arg), ... , args).finished();
      else if constexpr (rows != dynamic_size and columns == dynamic_size)
        return ((Eigen3::eigen_matrix_t<Scalar, rows, (1 + sizeof...(Args)) / rows> {} << arg), ... , args).finished();
      else if constexpr (rows == dynamic_size and columns != dynamic_size)
        return ((Eigen3::eigen_matrix_t<Scalar, (1 + sizeof...(Args)) / columns, columns> {} << arg), ... , args).finished();
      else
      {
        static_assert(rows == dynamic_size and columns == dynamic_size);
        return ((Eigen3::eigen_matrix_t<Scalar, 1 + sizeof...(Args), 1> {} << arg), ... , args).finished();
      }
    }


#ifdef __cpp_concepts
    template<scalar_constant<CompileTimeStatus::unknown> C, typename...Ds> requires (sizeof...(Ds) == 2)
    static constexpr constant_matrix<CompileTimeStatus::unknown> auto
#else
    template<typename C, typename...Ds, std::enable_if_t<
      scalar_constant<C, CompileTimeStatus::unknown> and (sizeof...(Ds) == 2), int> = 0>
    static constexpr auto
#endif
    make_constant_matrix(C&& c, Ds&&...ds)
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      using N = dense_writable_matrix_t<T, Scalar, std::decay_t<Ds>...>;
      return N::Constant(static_cast<index_type>(get_dimension_size_of(std::forward<Ds>(ds)))..., std::forward<C>(c));
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
        constexpr index_type n {dimension_size_of_v<D>};
        return Eigen3::eigen_matrix_t<Scalar, n, n>::Identity();
      }
    }


    template<typename Arg, typename...Begin, typename...Size>
    static auto
    get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      static_assert(sizeof...(Begin) == 2 and sizeof...(Size) == 2);

      // \todo If Eigen implements Block for non-dense matrices, this can be changed to Eigen3::eigen_general.
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        using B = Eigen::Block<std::remove_reference_t<Arg>,
          static_cast<index_type_of_t<Arg>>(static_index_value<Size> ? static_index_value_of_v<Size> : Eigen::Dynamic)...>;

        if constexpr ((static_index_value<Size> and ...))
          return make_self_contained<Arg>(B(arg, std::get<0>(begin), std::get<1>(begin)));
        else
          return make_self_contained<Arg>(B(arg, std::get<0>(begin), std::get<1>(begin), std::get<0>(size), std::get<1>(size)));
      }
      else return make_self_contained(get_block(to_native_matrix(std::forward<Arg>(arg)), begin, size));
    }


    template<typename Arg, typename Block, typename...Begin>
    static constexpr Arg&
    set_block(Arg& arg, Block&& block, Begin...begin)
    {
      static_assert(sizeof...(Begin) == 2);
      if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        set_block(nested_matrix(arg), std::forward<Block>(block), begin...);
        return arg;
      }
      else
      {
        static_assert(Eigen3::eigen_dense_general<Arg>);

        if constexpr (Eigen3::eigen_block<Block>)
        {
          if (std::addressof(arg) == std::addressof(block.nestedExpression()) and
              std::get<0>(std::tuple{begin...}) == block.startRow() and std::get<1>(std::tuple{begin...}) == block.startCol())
            return arg;
        }

        using B = Eigen::Block<std::remove_reference_t<Arg>,
          static_cast<index_type_of_t<Arg>>(index_dimension_of_v<Block, 0>),
          static_cast<index_type_of_t<Arg>>(index_dimension_of_v<Block, 1>)>;

        if constexpr (not has_dynamic_dimensions<Block>)
          B(arg, begin...) = std::forward<Block>(block);
        else
          B(arg, begin..., get_index_dimension_of<0>(block), get_index_dimension_of<1>(block)) = std::forward<Block>(block);
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
    static decltype(auto) set_triangle(A&& a, B&& b)
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
        if constexpr (t == TriangleType::diagonal)
        {
          if constexpr (writable<A> and std::is_lvalue_reference_v<A>)
          {
            a.diagonal() = OpenKalman::diagonal_of(std::forward<B>(b));
            return std::forward<A>(a);
          }
          else
          {
            auto aw = make_dense_writable_matrix_from(std::forward<A>(a));
            aw.diagonal() = OpenKalman::diagonal_of(std::forward<B>(b));
            return aw;
          }
        }
        else
        {
          decltype(auto) aw = [](A&& a) -> decltype(auto) {
            if constexpr (writable<A> and std::is_lvalue_reference_v<A>) return std::forward<A>(a);
            else return make_dense_writable_matrix_from(std::forward<A>(a));
          }(std::forward<A>(a));

          auto tview = aw.template triangularView<t == TriangleType::upper ? Eigen::Upper : Eigen::Lower>();
          if constexpr (std::is_assignable_v<decltype(tview), B&&>)
            tview = std::forward<B>(b);
          else
            tview = to_native_matrix(std::forward<B>(b));
          return std::forward<decltype(aw)>(aw);
        }
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

  private:

    template<typename D, typename Arg>
    static constexpr decltype(auto)
    diagonal_of_impl(Arg&& arg)
    {
      auto diag {[](Arg&& arg){
        if constexpr (Eigen3::eigen_array_general<Arg>)
          return make_self_contained<Arg>(std::forward<Arg>(arg).matrix().diagonal());
        else // eigen_matrix_general<Arg>
          return make_self_contained<Arg>(std::forward<Arg>(arg).diagonal());
      }(std::forward<Arg>(arg))};
      using Diag = const std::decay_t<decltype(diag)>;
      static_assert(vector<Diag>);

      // If the result of .diagonal() has dynamic rows but the true dimension is known statically, wrap in FixedSizeAdapter.
      if constexpr (not dynamic_dimension<Diag, 0> or dynamic_index_descriptor<D>) return diag;
      else return internal::FixedSizeAdapter<Diag, std::decay_t<D>, Dimensions<1>> {std::move(diag)};
    }

  public:

#ifdef __cpp_concepts
    template<Eigen3::eigen_general<true> Arg>
    static constexpr vector decltype(auto)
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_general<Arg, true>, int> = 0>
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
        using Scalar = scalar_type_of_t<Arg>;
        decltype(auto) diag {nested_matrix(std::forward<Arg>(arg))}; //< must be nested_matrix(...) rather than .diagonal() because of const_cast
        using Diag = decltype(diag);
        using EigenTraits = Eigen::internal::traits<std::decay_t<Diag>>;
        constexpr index_type_of_t<Arg> rows = EigenTraits::RowsAtCompileTime;
        constexpr index_type_of_t<Arg> cols = EigenTraits::ColsAtCompileTime;

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
          auto d {make_dense_writable_matrix_from(std::forward<Diag>(diag))};
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
      else if constexpr (Eigen3::eigen_wrapper<Arg>)
      {
        if constexpr (Eigen3::eigen_general<nested_matrix_of_t<Arg>, true>)
          return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg)));
        else
          return diagonal_of_impl<decltype(*d)>(std::forward<Arg>(arg));
      }
      else if constexpr (Eigen3::eigen_Identity<Arg>)
      {
        constexpr std::size_t dim = dynamic_rows<Arg> ? column_dimension_of_v<Arg> : row_dimension_of_v<Arg>;
        if constexpr (dim == dynamic_size) return make_constant_matrix_like<Arg, Scalar, 1>(*d, Dimensions<1>{});
        else return make_constant_matrix_like<Arg, Scalar, 1>(Dimensions<dim>{}, Dimensions<1>{});
      }
      else
      {
        return diagonal_of_impl<decltype(*d)>(std::forward<Arg>(arg));
      };
    }


    template<typename Arg, typename Factor0, typename Factor1 = std::integral_constant<index_type_of_t<Arg>, 1>>
    static auto
    broadcast(Arg&& arg, const Factor0& factor0, const Factor1& factor1 = Factor1{})
    {
      using Index = index_type_of_t<Arg>;
      constexpr Index F0 = dynamic_index_value<Factor0> ? Eigen::Dynamic : static_index_value_of_v<Factor0>;
      constexpr Index F1 = dynamic_index_value<Factor1> ? Eigen::Dynamic : static_index_value_of_v<Factor1>;
      using R = Eigen::Replicate<std::decay_t<Arg>, F0, F1>;
      if constexpr (dynamic_index_value<Factor0> or dynamic_index_value<Factor1>)
        return R {std::forward<Arg>(arg), static_cast<Index>(factor0), static_cast<Index>(factor1)};
      else
        return R {std::forward<Arg>(arg)};
    }

  private:

    template<typename Arg, typename...Ds, std::size_t...Is>
    static constexpr bool is_replicatable(std::index_sequence<Is...>)
    {
      return (... and (
        dynamic_dimension<Arg, Is> or dynamic_index_descriptor<std::tuple_element_t<Is, std::tuple<Ds...>>> or
        dimension_size_of_index_is<Arg, Is, 1> or
        dimension_size_of_index_is<Arg, Is, dimension_size_of_v<std::tuple_element_t<Is, std::tuple<Ds...>>>>));
    }


    template<typename...Ds, typename Arg, std::size_t...I>
    static decltype(auto)
    replicate_arg_impl(const std::tuple<Ds...>& d_tup, Arg&& arg, std::index_sequence<I...>)
    {
      using R = Eigen::Replicate<std::decay_t<Arg>,
        (dimension_size_of_v<Ds> == dynamic_size or dynamic_dimension<Arg, I> ?
        Eigen::Dynamic : static_cast<index_type_of_t<Arg>>(dimension_size_of_v<Ds> / index_dimension_of_v<Arg, I>))...>;

      if constexpr (((dimension_size_of_v<Ds> != dynamic_size) and ...) and
        ((index_dimension_of_v<Arg, I> != dynamic_size) and ...))
      {
        if constexpr (((dimension_size_of_v<Ds> == index_dimension_of_v<Arg, I>) and ...))
          return std::forward<Arg>(arg);
        else
          return R {std::forward<Arg>(arg)};
      }
      else
      {
        auto ret = R {std::forward<Arg>(arg), static_cast<index_type_of_t<Arg>>(
          get_dimension_size_of(std::get<I>(d_tup)) / get_index_dimension_of<I>(arg))...};
        return ret;
      }
    }


    /**
     * \brief Replicate a tensor, if necessary, to fill another tensor of higher or equal rank.
     * \details If Arg already has a rank equal to <code>sizeof...(Ds)</ref> return the argument unchanged.
     * \param d_tup A tuple of index descriptors (of type Ds) defining the resulting tensor
     * \tparam Arg The lower-rank tensor.
     * \return A higher-rank tensor filled with Arg.
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

  public:

#ifdef __cpp_concepts
    template<typename...Ds, typename Operation, typename...Args> requires (sizeof...(Ds) <= 2) and (sizeof...(Args) <= 3) and
      (scalar_type<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>...>> or
      (sizeof...(Args) == 0 and
        (scalar_type<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>..., std::conditional_t<true, index_type_of_t<T>, Ds>...>> or
        scalar_type<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>..., index_type_of_t<T>>>)))
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<sizeof...(Ds) <= 2 and sizeof...(Args) <= 3 and
      (scalar_type<typename std::invoke_result<Operation&&, typename scalar_type_of<Args>::type...>::type> or (sizeof...(Args) == 0 and
        (scalar_type<typename std::invoke_result<Operation&&, typename scalar_type_of<Args>::type..., std::conditional_t<true, typename index_type_of<T>::type, Ds>...>::type> or
        scalar_type<typename std::invoke_result<Operation&&, typename scalar_type_of<Args>::type..., typename index_type_of<T>::type>::type>))), int> = 0>
#endif
    static auto
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      decltype(auto) op = Eigen3::native_operation(std::forward<Operation>(operation));
      using Op = decltype(op);

      if constexpr (sizeof...(Args) == 0)
      {
        using P = dense_writable_matrix_t<T, scalar_type_of_t<T>, Ds...>;
        return Eigen::CwiseNullaryOp<std::decay_t<Op>, P> {
          static_cast<index_type_of_t<P>>(get_dimension_size_of(std::get<0>(tup))),
          static_cast<index_type_of_t<P>>(get_dimension_size_of(std::get<1>(tup))),
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


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr auto
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      if constexpr (Eigen3::eigen_dense_general<Arg>)
      {
        decltype(auto) op = Eigen3::native_operation(std::forward<BinaryFunction>(b));
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
      auto s = make_self_contained<A, B>(std::forward<A>(a) + std::forward<B>(b));
      if constexpr ((dynamic_dimension<decltype(s), 0> and (not dynamic_dimension<A, 0> or not dynamic_dimension<B, 0>)) or
          (dynamic_dimension<decltype(s), 1> and (not dynamic_dimension<A, 1> or not dynamic_dimension<B, 1>)))
      {
        using S0 = index_descriptor_of_t<decltype(s), 0>; using S1 = index_descriptor_of_t<decltype(s), 1>;
        using A0 = index_descriptor_of_t<A, 0>; using A1 = index_descriptor_of_t<A, 1>;
        using B0 = index_descriptor_of_t<B, 0>; using B1 = index_descriptor_of_t<B, 1>;

        using R = std::conditional_t<dynamic_index_descriptor<S0>,
          std::conditional_t<dynamic_index_descriptor<A0>, std::conditional_t<dynamic_index_descriptor<B0>, S0, B0>, A0>, S0>;
        using C = std::conditional_t<dynamic_index_descriptor<S1>,
          std::conditional_t<dynamic_index_descriptor<A1>, std::conditional_t<dynamic_index_descriptor<B1>, S1, B1>, A1>, S1>;
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
      decltype(auto) ma = [](A& a){
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
      using M = dense_writable_matrix_t<A, Scalar>;

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
          auto othercols = make_zero_matrix_like<A>(get_index_descriptor<0>(a), get_index_descriptor<0>(a) - 1);
          return TriangularMatrix<M, triangle_type> {concatenate_horizontal(col0, othercols)};
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto row0 = make_constant_matrix_like<A>(square_root(s), Dimensions<1>{}, Dimensions<dim>{});
          auto otherrows = make_zero_matrix_like<A>(get_index_descriptor<0>(a) - 1, get_index_descriptor<0>(a));
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

      constexpr std::size_t a_rows = dynamic_rows<A> ? row_dimension_of_v<B> : row_dimension_of_v<A>;
      constexpr std::size_t a_cols = column_dimension_of_v<A>;
      constexpr std::size_t b_cols = column_dimension_of_v<B>;

      if constexpr (not Eigen3::eigen_matrix_general<A, true>)
      {
        decltype(auto) n = OpenKalman::to_native_matrix(std::forward<A>(a));
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
        constexpr auto rows = row_dimension_of_v<A>;
        constexpr auto cols = column_dimension_of_v<A>;
        using MatrixType = Eigen3::eigen_matrix_t<Scalar, rows, cols>;
        using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;

        Eigen::HouseholderQR<MatrixType> QR {std::forward<A>(a)};

        if constexpr (dynamic_columns<A>)
        {
          auto rt_cols = get_index_dimension_of<1>(a);

          ResultType ret {rt_cols, rt_cols};

          if constexpr (dynamic_rows<A>)
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

          if constexpr (dynamic_rows<A>)
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

#endif //OPENKALMAN_EIGEN_GENERAL_LIBRARYROUTINES_HPP
