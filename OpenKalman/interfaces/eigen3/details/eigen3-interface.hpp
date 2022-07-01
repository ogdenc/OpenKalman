/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded functions relating to various Eigen3 types
 */

#ifndef OPENKALMAN_EIGEN3_INTERFACE_HPP
#define OPENKALMAN_EIGEN3_INTERFACE_HPP

#include <type_traits>
#include <tuple>
#include <random>
#include <special-matrices/TriangularMatrix.hpp>


namespace OpenKalman::interface
{
  namespace EI = Eigen::internal;

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct eigen_has_linear_access : std::false_type {};

    template<typename T>
    struct eigen_has_linear_access<T, std::enable_if_t<native_eigen_matrix<T> or native_eigen_array<T>>>
      : std::bool_constant<(Eigen::internal::evaluator<T>::Flags & Eigen::LinearAccessBit) != 0> {};
  }
#endif


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const int&>...I> requires
    (native_eigen_matrix<T> or native_eigen_array<T>) and (sizeof...(I) <= 2) and
    (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<T>>::Flags & Eigen::LinearAccessBit) != 0)
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<(native_eigen_matrix<T> or native_eigen_array<T>) and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const int&>) and
    (sizeof...(I) != 1 or detail::eigen_has_linear_access<std::decay_t<T>>::value)>, I...>
#endif
  {
    template<typename Arg>
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      return std::forward<Arg>(arg).coeff(static_cast<int>(i)...);
    }
  };


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const int&>...I> requires
    (eigen_SelfAdjointView<T> or eigen_TriangularView<T>) and (sizeof...(I) == 2)
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<(eigen_SelfAdjointView<T> or eigen_TriangularView<T>) and
    ((sizeof...(I) == 2) and ... and std::is_convertible_v<I, const int&>)>, I...>
#endif
  {
    template<typename Arg>
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      return std::forward<Arg>(arg).coeff(static_cast<int>(i)...);
    }
  };


#ifdef __cpp_concepts
  template<typename T, typename I> requires (eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    element_gettable<nested_matrix_of_t<T>, I>
  struct GetElement<T, I> : GetElement<nested_matrix_of_t<T>, I> {};
#else
  template<typename T, typename I>
  struct GetElement<T, std::enable_if_t<(eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    element_gettable<nested_matrix_of_t<T>, I>>, I> : GetElement<nested_matrix_of_t<T>, void, I> {};
#endif


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const int&> I, std::convertible_to<const int&> J> requires
    (eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    (element_gettable<nested_matrix_of_t<T>, I> or element_gettable<nested_matrix_of_t<T>, I, J>)
  struct GetElement<T, I, J>
#else
  template<typename T, typename I, typename J>
  struct GetElement<T, std::enable_if_t<(eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    std::is_convertible_v<I, const int&> and std::is_convertible_v<J, const int&> and
    (element_gettable<nested_matrix_of_t<T>, I> or element_gettable<nested_matrix_of_t<T>, I, J>)>, I, J>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i, J j)
    {
      if (i == j)
      {
        if constexpr (element_gettable<nested_matrix_of_t<Arg>, I>)
          return std::forward<Arg>(arg).diagonal().coeff(static_cast<int>(i));
        else
          return std::forward<Arg>(arg).diagonal().coeff(static_cast<int>(i), static_cast<int>(i));
      }
      else
      {
        return scalar_type_of_t<Arg>(0);
      }
    }
  };


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const int&>...I> requires
    (native_eigen_matrix<T> or native_eigen_array<T>) and (sizeof...(I) <= 2) and
    (sizeof...(I) != 1 or static_cast<bool>(Eigen::internal::evaluator<std::decay_t<T>>::Flags & Eigen::LinearAccessBit)) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, std::enable_if_t<((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const int&>) and
    (sizeof...(I) != 1 or detail::eigen_has_linear_access<std::decay_t<T>>::value) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))>, I...>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, Scalar s, I...i)
    {
      arg.coeffRef(static_cast<int>(i)...) = s;
    }
  };


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const int&>...I> requires
    (eigen_SelfAdjointView<T> or eigen_TriangularView<T>) and (sizeof...(I) == 2) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, std::enable_if_t<(eigen_SelfAdjointView<T> or eigen_TriangularView<T>) and
    ((sizeof...(I) == 2) and ... and std::is_convertible_v<I, const int&>) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))>, I...>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, Scalar s, I...i)
    {
      arg.coeffRef(static_cast<int>(i)...) = s;
    }
  };


#ifdef __cpp_concepts
  template<typename T, typename I> requires (eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    element_settable<nested_matrix_of_t<T>, I>
  struct SetElement<T, I>
#else
  template<typename T, typename I>
  struct SetElement<T, std::enable_if_t<(eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    element_settable<nested_matrix_of_t<T>, I>>, I>
#endif
    : SetElement<nested_matrix_of_t<T>, I> {};


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const int&> I, std::convertible_to<const int&> J> requires
    (eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    (element_settable<nested_matrix_of_t<T>, I> or element_settable<nested_matrix_of_t<T>, I, J>)
  struct SetElement<T, I, J>
#else
  template<typename T, typename I, typename J>
  struct SetElement<T, std::enable_if_t<(eigen_DiagonalMatrix<T> or eigen_DiagonalWrapper<T>) and
    std::is_convertible_v<I, const int&> and std::is_convertible_v<J, const int&> and
    (element_settable<nested_matrix_of_t<T>, I> or element_settable<nested_matrix_of_t<T>, I, J>)>, I, J>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, I i, J j)
    {
      if (i == j)
      {
        if constexpr (element_settable<nested_matrix_of_t<Arg>, std::size_t>)
          set_element(arg.diagonal(), s, static_cast<int>(i));
        else
          set_element(arg.diagonal(), s, static_cast<int>(i), static_cast<int>(i));
      }
      else if (s != 0)
        throw std::out_of_range("Cannot set non-diagonal element of a diagonal matrix to a non-zero value.");
    }
  };


#ifdef __cpp_concepts
  template<native_eigen_general T>
  struct Subsets<T>
#else
  template<typename T>
  struct Subsets<T, std::enable_if_t<native_eigen_general<T>>>
#endif
  {
    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto) column(Arg&& arg, runtime_index_t...i)
    {
      if constexpr (native_eigen_matrix<Arg> or native_eigen_array<Arg>)
      {
        auto e_i = static_cast<Eigen::Index>((compile_time_index + ... + (i + ... + 0)));
        return std::forward<Arg>(arg).col(e_i);
      }
      else
      {
        column<compile_time_index...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)), i...);
      }
    }


    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto) row(Arg&& arg, runtime_index_t...i)
    {
      if constexpr (native_eigen_matrix<Arg> or native_eigen_array<Arg>)
      {
        auto e_i = static_cast<Eigen::Index>((compile_time_index + ... + (i + ... + 0)));
        return std::forward<Arg>(arg).row(e_i);
      }
      else
      {
        row<compile_time_index...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)), i...);
      }
    }
  };


#ifdef __cpp_concepts
  template<native_eigen_general T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<native_eigen_general<T>>>
#endif
  {

    template<typename Arg>
    static auto
    to_diagonal(Arg&& arg)
    {
      if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg> or
        eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
      {
        // In this case, arg will be a one-by-one matrix.
        if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1) throw std::invalid_argument {
          "Argument of to_diagonal must have 1 column; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};
        return std::forward<Arg>(arg).nestedExpression();
      }
      else
      {
        return DiagonalMatrix<passable_t<Arg>> {std::forward<Arg>(arg)};
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    diagonal_of(Arg&& arg)
    {
      if constexpr (not square_matrix<Arg>) if (get_index_dimension_of<0>(arg) != get_index_dimension_of<1>(arg))
        throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead it has " +
          std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
          std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

      using Scalar = scalar_type_of_t<Arg>;

      constexpr std::size_t dim = dynamic_rows<Arg> ? column_dimension_of_v<Arg> : row_dimension_of_v<Arg>;

      if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
      {
        // Note: we assume that the nested matrix reference is not dangling.
        return diagonal_of(std::forward<Arg>(arg).nestedExpression());
      }
      else if constexpr (dim == 1 and (native_eigen_matrix<Arg> or native_eigen_array<Arg>))
      {
        return std::forward<Arg>(arg);
      }
      else if constexpr (native_eigen_matrix<Arg> or eigen_DiagonalMatrix<Arg>)
      {
        if constexpr (std::is_lvalue_reference_v<Arg> or self_contained<decltype(std::forward<Arg>(arg).diagonal())>)
        {
          return std::forward<Arg>(arg).diagonal();
        }
        else
        {
          auto d = std::forward<Arg>(arg).diagonal();
          return untyped_dense_writable_matrix_t<decltype(d), dim, 1> {std::move(d)};
        }
      }
      else if constexpr (native_eigen_array<Arg>)
      {
        if constexpr (std::is_lvalue_reference_v<Arg> or
          self_contained<decltype(std::forward<Arg>(arg).matrix().diagonal())>)
        {
          return std::forward<Arg>(arg).matrix().diagonal();
        }
        else
        {
          auto d = std::forward<Arg>(arg).matrix().diagonal();
          return untyped_dense_writable_matrix_t<decltype(d), dim, 1> {std::move(d)};
        }
      }
      else
      {
        static_assert(eigen_DiagonalWrapper<Arg>);
        auto& diag = std::forward<Arg>(arg).diagonal();
        using Diag = std::decay_t<decltype(diag)>;
        constexpr Eigen::Index rows = Eigen::internal::traits<Diag>::RowsAtCompileTime;
        constexpr Eigen::Index cols = Eigen::internal::traits<Diag>::ColsAtCompileTime;

        if constexpr (cols == 1 or cols == 0)
        {
          return (diag);
        }
        else if constexpr (rows == 1 or rows == 0)
        {
          return diag.transpose();
        }
        else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
        {
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<Diag>, Diag>)
          {
            return M::Map(diag.data(), (Eigen::Index) (get_index_dimension_of<0>(diag) *
              get_index_dimension_of<1>(diag)));
          }
          else
          {
            auto d = make_dense_writable_matrix_from(diag);
            return M::Map(d.data(), (Eigen::Index) (get_dimension_size_of(get_dimensions_of<0>(diag)) *
              get_dimension_size_of(get_dimensions_of<1>(diag))));
          }
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<Diag>, Diag>)
          {
            return M::Map(diag.data());
          }
          else
          {
            auto d = make_dense_writable_matrix_from(diag);
            return M::Map(d.data());
          }
        }
      }
    }

  };


#ifdef __cpp_concepts
  template<native_eigen_general T>
  struct ArrayOperations<T>
#else
  template<typename T>
  struct ArrayOperations<T, std::enable_if_t<native_eigen_general<T>>>
#endif
  {
  private:

    // Convert std functions to equivalent Eigen operations for possible vectorization:
    template<typename Op> static decltype(auto) nat_op(Op&& op) { return std::forward<Op>(op); };
    template<typename S> static auto nat_op(const std::plus<S>& op) { return EI::scalar_sum_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::minus<S>& op) { return EI::scalar_difference_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::multiplies<S>& op) {return EI::scalar_product_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::divides<S>& op) { return EI::scalar_quotient_op<S, S> {}; };
    template<typename S> static auto nat_op(const std::negate<S>& op) { return EI::scalar_opposite_op<S> {}; };

    using EIC = EI::ComparisonName;
    template<typename S> static auto nat_op(const std::equal_to<S>& op) { return EI::scalar_cmp_op<S, S, EIC::cmp_EQ> {}; };
    template<typename S> static auto nat_op(const std::not_equal_to<S>& op) { return EI::scalar_cmp_op<S, S, EIC::cmp_NEQ> {}; };
    template<typename S> static auto nat_op(const std::greater<S>& op) { return EI::scalar_cmp_op<S, S, EIC::cmp_GT> {}; };
    template<typename S> static auto nat_op(const std::less<S>& op) { return EI::scalar_cmp_op<S, S, EIC::cmp_LT> {}; };
    template<typename S> static auto nat_op(const std::greater_equal<S>& op) { return EI::scalar_cmp_op<S, S, EIC::cmp_GE> {}; };
    template<typename S> static auto nat_op(const std::less_equal<S>& op) { return EI::scalar_cmp_op<S, S, EIC::cmp_LE> {}; };

    template<typename S> static auto nat_op(const std::logical_and<S>& op) { return EI::scalar_boolean_and_op {}; };
    template<typename S> static auto nat_op(const std::logical_or<S>& op) { return EI::scalar_boolean_or_op {}; };
    template<typename S> static auto nat_op(const std::logical_not<S>& op) { return EI::scalar_boolean_not_op<S> {}; };


    template<typename...Ds, typename...ArgDs, typename Arg, std::size_t...I>
    static decltype(auto)
    replicate_arg_impl(const std::tuple<Ds...>& p_tup, const std::tuple<ArgDs...>& arg_tup, Arg&& arg, std::index_sequence<I...>)
    {
      using R = Eigen::Replicate<std::decay_t<Arg>,
        (dimension_size_of_v<Ds> == dynamic_size or dimension_size_of_v<ArgDs> == dynamic_size ?
        Eigen::Dynamic : static_cast<Eigen::Index>(dimension_size_of_v<Ds> / dimension_size_of_v<ArgDs>))...>;

      if constexpr (((dimension_size_of_v<Ds> != dynamic_size) and ...) and
        ((dimension_size_of_v<ArgDs> != dynamic_size) and ...))
      {
        if constexpr (((dimension_size_of_v<Ds> == dimension_size_of_v<ArgDs>) and ...))
          return std::forward<Arg>(arg);
        else
          return R {std::forward<Arg>(arg)};
      }
      else
      {
        auto ret = R {std::forward<Arg>(arg), static_cast<Eigen::Index>(
          get_dimension_size_of(std::get<I>(p_tup)) / get_dimension_size_of(std::get<I>(arg_tup)))...};
        return ret;
      }
    }


    template<typename...Ds, typename Arg>
    static decltype(auto)
    replicate_arg(const std::tuple<Ds...>& p_tup, Arg&& arg)
    {
      return replicate_arg_impl(p_tup, get_all_dimensions_of(arg), std::forward<Arg>(arg),
        std::make_index_sequence<sizeof...(Ds)>{});
    }

  public:

#ifdef __cpp_concepts
    template<typename...Ds, typename Operation, typename...Args> requires (sizeof...(Args) <= 3) and
      std::invocable<Operation&&, scalar_type_of_t<Args>...> and
      scalar_type<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>...>>
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<sizeof...(Args) <= 3 and
      std::is_invocable<Operation&&, scalar_type_of_t<Args>...>::value and
      scalar_type<std::invoke_result<Operation&&, scalar_type_of_t<Args>...>, int> = 0>
#endif
    static auto
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& operation, Args&&...args)
    {
      decltype(auto) op = nat_op(std::forward<Operation>(operation));
      using Op = decltype(op);

      if constexpr (sizeof...(Args) == 0)
      {
        using P = dense_writable_matrix_t<T, Ds...>;
        Eigen::Index r = get_dimension_size_of(std::get<0>(tup));
        Eigen::Index c = get_dimension_size_of(std::get<1>(tup));
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


#ifdef __cpp_concepts
    template<typename...Ds, typename Operation>
    requires std::invocable<Operation&&, typename dimension_size_of<Ds>::type...> and
      scalar_type<std::invoke_result_t<Operation&&, typename dimension_size_of<Ds>::type...>>
#else
    template<typename...Ds, typename Operation, std::enable_if_t<
      std::is_invocable<Operation&&, typename dimension_size_of<Ds>::type...>::value and
      scalar_type<std::invoke_result<Operation&&, typename dimension_size_of<Ds>::type...>::type>, int> = 0>
#endif
    static constexpr auto
    n_ary_operation_with_indices(const std::tuple<Ds...>& d_tup, Operation&& operation)
    {
      using P = dense_writable_matrix_t<T, Ds...>;
      Eigen::Index r = get_dimension_size_of(std::get<0>(d_tup));
      Eigen::Index c = get_dimension_size_of(std::get<1>(d_tup));
      return Eigen::CwiseNullaryOp<std::decay_t<Operation>, P> {r, c, std::forward<Operation>(operation)};
    }

  private:

    template<typename U> struct is_plus : std::false_type {};
    template<typename U> struct is_plus<std::plus<U>> : std::true_type {};
    template<typename U> struct is_multiplies : std::false_type {};
    template<typename U> struct is_multiplies<std::multiplies<U>> : std::true_type {};

  public:

    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr auto
    reduce(const BinaryFunction& b, Arg&& arg)
    {
      decltype(auto) dense_arg = [](Arg&& arg) -> decltype(auto) {
        if constexpr ((native_eigen_matrix<Arg> or native_eigen_array<Arg>)) return std::forward<Arg>(arg);
        else return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg));

      using DenseArg = decltype(dense_arg);

      if constexpr (sizeof...(indices) == 2) // reduce in both directions
      {
          return std::forward<DenseArg>(dense_arg).redux(b);
      }
      else if constexpr (is_plus<BinaryFunction>::value)
      {
        if constexpr (((indices == 0) and ...))
          return make_self_contained<DenseArg>(std::forward<DenseArg>(dense_arg).colwise().sum());
        else
          return make_self_contained<DenseArg>(std::forward<DenseArg>(dense_arg).rowwise().sum());
      }
      else if constexpr (is_multiplies<BinaryFunction>::value)
      {
        if constexpr (((indices == 0) and ...))
          return make_self_contained<DenseArg>(std::forward<DenseArg>(dense_arg).colwise().prod());
        else
          return make_self_contained<DenseArg>(std::forward<DenseArg>(dense_arg).rowwise().prod());
      }
      else
      {
        using OpWrapper = Eigen::internal::member_redux<BinaryFunction, scalar_type_of_t<Arg>>;
        constexpr auto dir = ((indices == 0) and ...) ? Eigen::Vertical : Eigen::Horizontal;
        using P = Eigen::PartialReduxExpr<std::decay_t<DenseArg>, OpWrapper, dir>;
        return make_self_contained<DenseArg>(P {std::forward<DenseArg>(dense_arg), OpWrapper {b}});
      }
    }

  };


#ifdef __cpp_concepts
  template<native_eigen_general T>
  struct ModularTransformationTraits<T>
#else
  template<typename T>
  struct ModularTransformationTraits<T, std::enable_if_t<native_eigen_general<T>>>
#endif
  {
    template<typename Arg, typename C>
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, const C& c) noexcept
    {
      return ToEuclideanExpr<C, Arg>(std::forward<Arg>(arg), c);
    }


    template<typename Arg, typename C>
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, const C& c) noexcept
    {
      return FromEuclideanExpr<C, Arg>(std::forward<Arg>(arg), c);
    }


    template<typename Arg, typename C>
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, const C& c) noexcept
    {
      return from_euclidean(to_euclidean(std::forward<Arg>(arg), c), c);
    }

  };


#ifdef __cpp_concepts
  template<native_eigen_general T>
  struct LinearAlgebra<T>
#else
  template<typename T>
  struct LinearAlgebra<T, std::enable_if_t<native_eigen_general<T>>>
#endif
  {

    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg) noexcept
    {
      if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
      {
        return make_self_contained<Arg>(to_diagonal(std::forward<Arg>(arg).diagonal().conjugate()));
      }
      else
      {
        return std::forward<Arg>(arg).conjugate();
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg) noexcept
    {
      // The global transpose function already handles symmetric or square constant matrices.
      if constexpr (zero_matrix<Arg>)
      {
        return OpenKalman::transpose(ZeroMatrix {std::forward<Arg>(arg)});
      }
      else if constexpr (constant_matrix<Arg>)
      {
        return OpenKalman::transpose(ConstantMatrix {std::forward<Arg>(arg)});
      }
      else if constexpr (triangular_matrix<Arg>)
      {
        return OpenKalman::transpose(TriangularMatrix {std::forward<Arg>(arg)});
      }
      else
      {
        static_assert(native_eigen_matrix<Arg> or native_eigen_array<Arg> or eigen_SelfAdjointView<Arg>);
        return std::forward<Arg>(arg).transpose();
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg) noexcept
    {
      // The global adjoint function already handles non-complex, hermitian, constant, and constant-diagonal cases.
      if constexpr (constant_matrix<Arg>)
      {
        return OpenKalman::adjoint(ConstantMatrix {std::forward<Arg>(arg)});
      }
      else if constexpr (triangular_matrix<Arg>)
      {
        return OpenKalman::adjoint(TriangularMatrix {std::forward<Arg>(arg)});
      }
      else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
      {
        return make_self_contained<Arg>(to_diagonal(std::forward<Arg>(arg).diagonal().conjugate()));
      }
      else
      {
        static_assert(native_eigen_matrix<Arg> or native_eigen_array<Arg> or eigen_SelfAdjointView<Arg>);
        return std::forward<Arg>(arg).adjoint();
      }
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg) noexcept
    {
      if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
      {
        return OpenKalman::determinant(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
      }
      else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
      {
        return std::forward<Arg>(arg).diagonal().prod();
      }
      else
      {
        return arg.determinant();
      }
    }


    template<typename Arg>
    static constexpr auto
    trace(Arg&& arg) noexcept
    {
      if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
      {
        return std::forward<Arg>(arg).nestedExpression().trace();
      }
      else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
      {
        return std::forward<Arg>(arg).diagonal().sum();
      }
      else
      {
        return arg.trace();
      }
    }


    template<TriangleType t, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      // Diagonal or one-by-one matrices:
      if constexpr (((t == TriangleType::diagonal or diagonal_matrix<A> or (dynamic_rows<A> and column_vector<A>) or
          (dynamic_columns<A> and row_vector<A>)) and diagonal_matrix<U>) or row_vector<U>)
      {
        // One-by-one matrices:
        if constexpr (one_by_one_matrix<A> or (dynamic_rows<A> and column_vector<A>) or
          (dynamic_columns<A> and row_vector<A>) or row_vector<U>)
        {
          auto e = OpenKalman::trace(a) + alpha * OpenKalman::trace(u) * trace(OpenKalman::conjugate(u));

          if constexpr (element_settable<A, std::size_t>)
          {
            set_element(a, e, 0);
            return std::forward<A>(a);
          }
          else if constexpr (element_settable<A, std::size_t, std::size_t>)
          {
            set_element(a, e, 0, 0);
            return std::forward<A>(a);
          }
          else if constexpr (std::is_assignable_v<A, decltype(MatrixTraits<A>::make(e))>)
          {
            a = MatrixTraits<A>::make(e);
            return std::forward<A>(a);
          }
          else
          {
            return MatrixTraits<A>::make(e);
          }
        }
        else // diagonal matrices
        {
          auto ud = diagonal_of(std::forward<U>(u));

          using Scalar = scalar_type_of_t<U>;
          using conj_prod = Eigen::internal::scalar_conj_product_op<Scalar, Scalar>;

          auto udprod = Eigen::CwiseBinaryOp<conj_prod, decltype(ud), decltype(ud)> {ud, ud};

          auto d = diagonal_of(a) + alpha * udprod;
          using D = decltype(d);

          if constexpr (native_eigen_matrix<D>)
          {
            if constexpr (std::is_assignable_v<A, decltype(d.asDiagonal())>)
            {
              a = d.asDiagonal();
              return std::forward<A>(a);
            }
          }

          if constexpr (std::is_assignable_v<A, decltype(d.matrix().asDiagonal())>)
          {
            a = d.matrix().asDiagonal();
            return std::forward<A>(a);
          }
          else if constexpr (eigen_DiagonalMatrix<A> and std::is_assignable_v<decltype(a.diagonal()), D>)
          {
            a.diagonal() = d;
            return std::forward<A>(a);
          }
          else if constexpr (std::is_constructible_v<std::decay_t<A>, D> and std::is_assignable_v<A, std::decay_t<A>>)
          {
            a = std::decay_t<A> {d};
            return std::forward<A>(a);
          }
          else
          {
            return to_diagonal(make_self_contained<A, U>(std::move(d)));
          }
        }
      }
      else // non-diagonal hermitian matrices
      {
        constexpr auto UpLo = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        constexpr auto s = t == TriangleType::upper ? t : TriangleType::lower;

        // Get the underlying writable, dense matrix:
        decltype(auto) aw = [](A&& a) -> decltype(auto) {
          if constexpr (eigen_SelfAdjointView<A>)
          {
            if constexpr (t == self_adjoint_triangle_type_of_v<A>)
              return make_dense_writable_matrix_from(std::forward<A>(a).nestedExpression());
            else
              return OpenKalman::adjoint(make_dense_writable_matrix_from(std::forward<A>(a).nestedExpression()));
          }
          else
          {
            return make_dense_writable_matrix_from(std::forward<A>(a));
          }
        }(std::forward<A>(a));

        // Make a copy of Aw if either A or Aw are not a non-const lvalue references:
        using Aw = decltype(aw);
        using Ax = std::conditional_t<
          std::is_lvalue_reference_v<A> and not std::is_const_v<std::remove_reference_t<A>> and
          std::is_lvalue_reference_v<Aw> and not std::is_const_v<std::remove_reference_t<Aw>>,
          Aw, std::decay_t<Aw>>;
        Ax ax {std::forward<Aw>(aw)};

        // Perform the rank update and construct a SelfAdjointMatrix wrapper:
        ax.template selfadjointView<UpLo>().template rankUpdate(std::forward<U>(u), alpha);
        return SelfAdjointMatrix<Ax, s> {std::forward<Ax>(ax)};
      }
    }

  private:

    template<int UpLo, typename Arg, typename U, typename Alpha>
    static Arg&
    rank_update_tri_impl(Arg& arg, U&& u, const Alpha alpha)
    {
      using Scalar = scalar_type_of_t<Arg>;

      decltype(auto) v = [](U&& u) -> decltype(auto) {
        if constexpr (native_eigen_matrix<U> or native_eigen_array<U>)
          return std::forward<U>(u);
        else
          return make_dense_writable_matrix_from(std::forward<U>(u));
      }(std::forward<U>(u));

      for (std::size_t i = 0; i < get_index_dimension_of<1>(v); i++)
      {
        if (Eigen::internal::llt_inplace<Scalar, UpLo>::rankUpdate(arg, column(v, i), alpha) >= 0)
          throw (std::runtime_error("rank_update_triangular: product is not positive definite"));
      }

      return arg;
    }

  public:

    template<TriangleType t, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      // A is diagonal or one-by-one and u is diagonal or a row vector:
      if constexpr (((t == TriangleType::diagonal or diagonal_matrix<A> or (dynamic_rows<A> and column_vector<A>) or
          (dynamic_columns<A> and row_vector<A>)) and diagonal_matrix<U>) or row_vector<U>)
      {
        // One-by-one matrices:
        if constexpr (one_by_one_matrix<A> or (dynamic_rows<A> and column_vector<A>) or
          (dynamic_columns<A> and row_vector<A>) or row_vector<U>)
        {
          auto e = std::sqrt(OpenKalman::trace(a) * OpenKalman::trace(OpenKalman::conjugate(a)) +
            alpha * OpenKalman::trace(u) * OpenKalman::trace(OpenKalman::conjugate(u)));

          if constexpr (element_settable<A, std::size_t>)
          {
            set_element(a, e, 0);
            return std::forward<A>(a);
          }
          else if constexpr (element_settable<A, std::size_t, std::size_t>)
          {
            set_element(a, e, 0, 0);
            return std::forward<A>(a);
          }
          else if constexpr (std::is_assignable_v<A, decltype(MatrixTraits<A>::make(e))>)
          {
            a = MatrixTraits<A>::make(e);
            return std::forward<A>(a);
          }
          else
          {
            return MatrixTraits<A>::make(e);
          }
        }
        else // diagonal matrices
        {
          auto ad = diagonal_of(a);
          auto ud = diagonal_of(std::forward<U>(u));

          using Scalar = scalar_type_of_t<U>;
          using conj_prod = Eigen::internal::scalar_conj_product_op<Scalar, Scalar>;

          auto adprod = Eigen::CwiseBinaryOp<conj_prod, decltype(ad), decltype(ad)> {ad, ad};
          auto udprod = Eigen::CwiseBinaryOp<conj_prod, decltype(ud), decltype(ud)> {ud, ud};

          auto d = (adprod.array() + alpha * udprod.array()).sqrt().matrix(); // decltype(d) is native_eigen_matrix
          using D = decltype(d);

          if constexpr (std::is_assignable_v<A, decltype(d.asDiagonal())>)
          {
            a = d.asDiagonal();
            return std::forward<A>(a);
          }
          else if constexpr (eigen_DiagonalMatrix<A> and std::is_assignable_v<decltype(a.diagonal()), D>)
          {
            a.diagonal() = d;
            return std::forward<A>(a);
          }
          else if constexpr (std::is_constructible_v<std::decay_t<A>, D> and std::is_assignable_v<A, std::decay_t<A>>)
          {
            a = std::decay_t<A> {d};
            return std::forward<A>(a);
          }
          else
          {
            return to_diagonal(make_self_contained<A, U>(std::move(d)));
          }
        }
      }
      else // non-diagonal, triangular matrices:
      {
        constexpr auto UpLo = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        constexpr auto s = t == TriangleType::upper ? t : TriangleType::lower;

        if constexpr (std::is_lvalue_reference_v<A> and not std::is_const_v<std::remove_reference_t<A>> and
          writable<A> and not diagonal_matrix<A>)
        {
          if constexpr (eigen_TriangularView<A>)
          {
            rank_update_tri_impl<UpLo>(a.nestedExpression(), std::forward<U>(u), alpha);
            return std::forward<A>(a);
          }
          else
          {
            return rank_update_tri_impl<UpLo>(a, std::forward<U>(u), alpha);
          }
        }
        //else if constexpr (zero_matrix<A>) //\todo Add this back in after revising interface for decompositions
        //{
        //  if constexpr (t == TriangleType::upper)
        //    return QR_decomposition(std::sqrt(alpha) * adjoint(u));
        //  else
        //    return LQ_decomposition(std::sqrt(alpha) * u);
        //}
        else // arg is not directly updatable and must be copied or moved
        {
          auto aw = make_dense_writable_matrix_from<A>(std::forward<A>(a));
          rank_update_tri_impl<UpLo>(aw, std::forward<U>(u), alpha);
          return TriangularMatrix<decltype(aw), s> {std::move(aw)};
        }
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

      if constexpr(not native_eigen_matrix<A>)
      {
        decltype(auto) n = to_native_matrix(std::forward<A>(a));
        static_assert(native_eigen_matrix<decltype(n)>);
        return solve<must_be_unique, must_be_exact>(std::forward<decltype(n)>(n), std::forward<B>(b));
      }
      else if constexpr (triangular_matrix<A>)
      {
        constexpr auto uplo = triangle_type_of_v<A> == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        return make_self_contained<A, B>(
          Eigen::Solve {std::forward<A>(a).template triangularView<uplo>(), std::forward<B>(b)});
      }
      else if constexpr (self_adjoint_matrix<A>)
      {
        constexpr auto uplo = self_adjoint_triangle_type_of_v<A> == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
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
          Eigen::ColPivHouseholderQR<eigen_matrix_t<Scalar, a_rows, a_cols>> QR {std::forward<A>(a)};
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
          Eigen::HouseholderQR<eigen_matrix_t<Scalar, a_rows, a_cols>> QR {std::forward<A>(a)};
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
      using Scalar = scalar_type_of_t<A>;
      constexpr auto rows = row_dimension_of_v<A>;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, rows, rows>;
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::lower>;

      ResultType ret = adjoint(QR_decomp_impl(adjoint(std::forward<A>(a))));

      return MatrixTraits<TType>::make(std::move(ret));
    }


    template<typename A>
    static constexpr auto
    QR_decomposition(A&& a)
    {
      using Scalar = scalar_type_of_t<A>;
      constexpr auto cols = column_dimension_of_v<A>;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::upper>;

      ResultType ret = QR_decomp_impl(std::forward<A>(a));

      return MatrixTraits<TType>::make(std::move(ret));
    }

  };


} // namespace OpenKalman::interface


namespace OpenKalman::Eigen3
{

  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects vertically.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    ((dynamic_columns<V> or dynamic_columns<Vs> or column_dimension_of<V>::value == column_dimension_of<Vs>::value) and ...)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = scalar_type_of_t<V>;

      if constexpr ((dynamic_columns<V> and ... and dynamic_columns<Vs>))
      {
        auto cols = get_dimensions_of<1>(v);
        assert(((cols == get_dimensions_of<1>(vs)) and ...));

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          auto rows = (get_dimensions_of<0>(v) + ... + get_dimensions_of<0>(vs));
          Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto rows = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
          Eigen3::eigen_matrix_t<Scalar, rows, dynamic_size> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
      else
      {
        constexpr auto cols = std::max({
          dynamic_columns<V> ? 0 : column_dimension_of_v<V>, dynamic_columns<Vs> ? 0 : column_dimension_of_v<Vs>...});

        static_assert(((dynamic_columns<V> or column_dimension_of_v<V> == cols) and ... and
          (dynamic_columns<Vs> or column_dimension_of_v<Vs> == cols)));

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          auto rows = (get_dimensions_of<0>(v) + ... + get_dimensions_of<0>(vs));
          Eigen3::eigen_matrix_t<Scalar, dynamic_size, cols> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto rows = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
          Eigen3::eigen_matrix_t<Scalar, rows, cols> m;
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects horizontally.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    ((dynamic_rows<V> or dynamic_rows<Vs> or row_dimension_of<V>::value == row_dimension_of<Vs>::value) and ...)
#ifndef __cpp_concepts
      , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = scalar_type_of_t<V>;

      if constexpr ((dynamic_rows<V> and ... and dynamic_rows<Vs>))
      {
        auto rows = get_dimensions_of<0>(v);
        assert(((rows == get_dimensions_of<0>(vs)) and ...));

        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto cols = (get_dimensions_of<1>(v) + ... + get_dimensions_of<1>(vs));
          Eigen3::eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto cols = (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>);
          Eigen3::eigen_matrix_t<Scalar, dynamic_size, cols> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
      else
      {
        constexpr auto rows = std::max({
          dynamic_rows<V> ? 0 : row_dimension_of_v<V>, dynamic_rows<Vs> ? 0 : row_dimension_of_v<Vs>...});

        static_assert(((dynamic_rows<V> or row_dimension_of_v<V> == rows) and ... and
          (dynamic_rows<Vs> or row_dimension_of_v<Vs> == rows)));

        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto cols = (get_dimensions_of<1>(v) + ... + get_dimensions_of<1>(vs));
          Eigen3::eigen_matrix_t<Scalar, rows, dynamic_size> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto cols = (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>);
          Eigen3::eigen_matrix_t<Scalar, rows, cols> m;
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace detail
  {
    // Concatenate one or more Eigen::MatrixBase objects diagonally.
    template<typename M, typename ... Vs, std::size_t ... ints>
    void concatenate_diagonal_impl(M& m, const std::tuple<Vs...>& vs_tup, std::index_sequence<ints...>)
    {
      constexpr auto dim = sizeof...(Vs);

      ((m << std::get<0>(vs_tup)), ..., [] (const auto& vs_tup)
      {
        constexpr auto row = (ints + 1) / dim;
        constexpr auto col = (ints + 1) % dim;
        using Vs_row = std::tuple_element_t<row, std::tuple<Vs...>>;
        using Vs_col = std::tuple_element_t<col, std::tuple<Vs...>>;

        if constexpr (row == col)
          return std::get<row>(vs_tup);
        else
          return make_zero_matrix_like<decltype(std::get<row>(vs_tup))>(
            Dimensions<row_dimension_of_v<Vs_row>>{}, Dimensions<column_dimension_of_v<Vs_col>>{});
      }(vs_tup));
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects diagonally.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (diagonal_matrix<V> and ... and diagonal_matrix<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    (not (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>)) and
    (not (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>))
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = scalar_type_of_t<V>;
      auto seq = std::make_index_sequence<(sizeof...(vs) + 1) * (sizeof...(vs) + 1) - 1> {};

      if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
      {
        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto rows = (get_dimensions_of<0>(v) + ... + get_dimensions_of<0>(vs));
          auto cols = (get_dimensions_of<1>(v) + ... + get_dimensions_of<1>(vs));
          eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
        else
        {
          auto rows = (get_dimensions_of<0>(v) + ... + get_dimensions_of<0>(vs));
          constexpr auto cols = (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>);
          eigen_matrix_t<Scalar, dynamic_size, cols> m {rows, cols};
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
      }
      else
      {
        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          constexpr auto rows = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
          auto cols = (get_dimensions_of<1>(v) + ... + get_dimensions_of<1>(vs));
          eigen_matrix_t<Scalar, rows, dynamic_size> m {rows, cols};
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
        else
        {
          constexpr auto rows = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
          constexpr auto cols = (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>);
          eigen_matrix_t<Scalar, rows, cols> m;
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace detail
  {
    /// Make a tuple containing an Eigen matrix (general case).
    template<typename F, typename RC, typename CC, typename Arg>
    auto
    make_split_tuple(Arg&& arg)
    {
      auto val = F::template call<RC, CC>(std::forward<Arg>(arg));
      return std::tuple<const decltype(val)> {std::move(val)};
    }


    /// Make a tuple containing an Eigen::Block.
    template<typename F, typename RC, typename CC, typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto
    make_split_tuple(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& arg)
    {
      auto b = [](auto& arg) {
        using NonConstBlock = Eigen::Block<std::remove_const_t<XprType>, BlockRows, BlockCols, InnerPanel>;

        // A const_cast is necessary, because a const Eigen::Block cannot be inserted into a tuple.
        auto& xpr = const_cast<std::remove_const_t<XprType>&>(arg.nestedExpression());

        if constexpr (BlockRows == Eigen::Dynamic or BlockCols == Eigen::Dynamic)
          return NonConstBlock(xpr, arg.startRow(), arg.startCol(), get_dimensions_of<0>(arg), get_dimensions_of<1>(arg));
        else
          return NonConstBlock(xpr, arg.startRow(), arg.startCol());
      } (arg);

      auto val = F::template call<RC, CC>(std::move(b));
      return std::tuple<const decltype(val)> {std::move(val)};
    }

  }


  /**
   * \brief Split a matrix vertically.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam euclidean Whether coefficients RC and RCs are transformed to Euclidean space.
   * \tparam RC TypedIndex for the first cut.
   * \tparam RCs TypedIndex for each of the second and subsequent cuts.
   * \todo add runtime-specified cuts
   */
#ifdef __cpp_concepts
  template<typename F, bool euclidean, typename RC, typename...RCs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or ((euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>) + ... +
    (euclidean ? euclidean_dimension_size_of_v<RCs> : dimension_size_of_v<RCs>)) <= row_dimension_of_v<Arg>)
#else
  template<typename F, bool euclidean, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_rows<Arg> or ((euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<RCs> : dimension_size_of_v<RCs>)) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg>)
        return std::forward<decltype(m)>(m);
      else
        return make_dense_writable_matrix_from(std::forward<decltype(m)>(m));
    };

    using CC = Dimensions<dynamic_columns<Arg> ? 0 : column_dimension_of_v<Arg>>; // \todo fix this
    constexpr Eigen::Index dim1 = euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>;

    if constexpr (sizeof...(RCs) > 0)
    {
      if constexpr (dynamic_rows<Arg>)
      {
        Eigen::Index dim2 = get_dimensions_of<0>(arg) - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.topRows(dim1))),
          split_vertical<F, euclidean, RCs...>(g(arg.bottomRows(dim2))));
      }
      else
      {
        constexpr Eigen::Index dim2 = row_dimension_of_v<Arg> - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>())),
          split_vertical<F, euclidean, RCs...>(g(arg.template bottomRows<dim2>())));
      }
    }
    else if constexpr (dim1 < row_dimension_of_v<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
        return detail::make_split_tuple<F, RC, CC>(g(arg.topRows(dim1)));
      else
        return detail::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Split a matrix vertically (case in which there is no split).
   */
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return std::tuple {};
  }


  /**
   * \brief Split a matrix vertically.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam RCs TypedIndex for each of the cuts.
   */
#ifdef __cpp_concepts
  template<typename F, typed_index_descriptor RC, typed_index_descriptor...RCs, eigen_matrix Arg>
  requires (not typed_index_descriptor<F>) and
  (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of_v<Arg>)
#else
  template<typename F, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    not typed_index_descriptor<F> and (typed_index_descriptor<RC> and ... and typed_index_descriptor<RCs>) and
      (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, false, RC, RCs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix vertically.
   * \tparam RC TypedIndex for the first cut.
   * \tparam RCs TypedIndex for the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typed_index_descriptor RC, typed_index_descriptor...RCs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of_v<Arg>)
#else
  template<typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (typed_index_descriptor<RC> and ... and typed_index_descriptor<RCs>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<OpenKalman::internal::default_split_function, RC, RCs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix vertically.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Numbers of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and
      (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix horizontally and invoke function F on each segment, returning a tuple.
   * \tparam CC TypedIndex for the first cut.
   * \tparam CCs TypedIndex for each of the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
#ifdef __cpp_concepts
  template<typename F, typed_index_descriptor CC, typed_index_descriptor...CCs, eigen_matrix Arg>
  requires (not typed_index_descriptor<F>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and not typed_index_descriptor<F> and
      (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_dense_writable_matrix_from(std::forward<decltype(m)>(m));
    };

    using RC = Dimensions<dynamic_rows<Arg> ? 0 : row_dimension_of_v<Arg>>; // \todo fix this
    constexpr Eigen::Index dim1 = dimension_size_of_v<CC>;

    if constexpr(sizeof...(CCs) > 0)
    {
      if constexpr (dynamic_columns<Arg>)
      {
        Eigen::Index dim2 = get_dimensions_of<1>(arg) - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.leftCols(dim1))),
          split_horizontal<F, CCs...>(g(arg.rightCols(dim2))));
      }
      else
      {
        constexpr Eigen::Index dim2 = column_dimension_of_v<Arg> - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>())),
          split_horizontal<F, CCs...>(g(arg.template rightCols<dim2>())));
      }
    }
    else if constexpr (dim1 < column_dimension_of_v<Arg>)
    {
      if constexpr (dynamic_columns<Arg>)
        return detail::make_split_tuple<F, RC, CC>(g(arg.leftCols(dim1)));
      else
        return detail::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Split a matrix horizontally (case in which there is no split).
   */
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, typename Arg,
    std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }


  /**
   * \brief Split a matrix horizontally.
   * \tparam CC TypedIndex for the first cut.
   * \tparam CCs TypedIndex for the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typed_index_descriptor CC, typed_index_descriptor...CCs, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of_v<Arg>)
#else
  template<typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and typed_index_descriptor<CC> and
      (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, CC, CCs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix horizontally.
   * \tparam cut Number of columns in the first cut.
   * \tparam cuts Numbers of columns in the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix diagonally and invoke function F on each segment, returning a tuple.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   * \tparam C TypedIndex for the first cut.
   * \tparam Cs TypedIndex for each of the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typename F, bool euclidean, typed_index_descriptor C, typed_index_descriptor...Cs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or
    ((euclidean ? euclidean_dimension_size_of_v<C> : dimension_size_of_v<C>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<Cs> : dimension_size_of_v<Cs>)) <= row_dimension_of_v<Arg>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, bool euclidean, typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and (typed_index_descriptor<C> and ... and typed_index_descriptor<Cs>) and
    (dynamic_rows<Arg> or
    ((euclidean ? euclidean_dimension_size_of_v<C> : dimension_size_of_v<C>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<Cs> : dimension_size_of_v<Cs>)) <= row_dimension_of<Arg>::value) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_dense_writable_matrix_from(std::forward<decltype(m)>(m));
    };

    constexpr Eigen::Index rdim1 = euclidean ? euclidean_dimension_size_of_v<C> : dimension_size_of_v<C>;
    constexpr Eigen::Index cdim1 = dimension_size_of_v<C>;

    if constexpr(sizeof...(Cs) > 0)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
      {
        Eigen::Index rdim2 = get_dimensions_of<0>(arg) - rdim1;
        Eigen::Index cdim2 = get_dimensions_of<1>(arg) - cdim1;

        return std::tuple_cat(
          detail::make_split_tuple<F, C, C>(g(arg.topLeftCorner(rdim1, cdim1))),
          split_diagonal<F, euclidean, Cs...>(g(arg.bottomRightCorner(rdim2, cdim2))));
      }
      else
      {
        constexpr Eigen::Index rdim2 = row_dimension_of_v<Arg> - rdim1;
        constexpr Eigen::Index cdim2 = column_dimension_of_v<Arg> - cdim1;

        return std::tuple_cat(
          detail::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>())),
          split_diagonal<F, euclidean, Cs...>(g(arg.template bottomRightCorner<rdim2, cdim2>())));
      }
    }
    else if constexpr(rdim1 < std::decay_t<Arg>::RowsAtCompileTime)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
        return detail::make_split_tuple<F, C, C>(g(arg.topLeftCorner(rdim1, cdim1)));
      else
        return detail::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, C, C>(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Split a matrix diagonally (case in which there is no split).
   */
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
  requires (not typed_index_descriptor<F>)
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg> and not typed_index_descriptor<F>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam C TypedIndex for the first cut.
   * \tparam Cs TypedIndex for each of the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typename F, typed_index_descriptor C, typed_index_descriptor...Cs, eigen_matrix Arg>
  requires (not typed_index_descriptor<F>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of_v<Arg>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and not typed_index_descriptor<F> and (typed_index_descriptor<C> and ... and typed_index_descriptor<Cs>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of<Arg>::value) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, false, C, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam C TypedIndex for the first cut.
   * \tparam Cs TypedIndex for the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typed_index_descriptor C, typed_index_descriptor...Cs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of_v<Arg>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and (typed_index_descriptor<C> and ... and typed_index_descriptor<Cs>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of<Arg>::value) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<OpenKalman::internal::default_split_function, C, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam cut Number of rows and columns in the first cut.
   * \tparam cuts Numbers of rows and columns in the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of_v<Arg>) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of<Arg>::value) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }


  // \todo Add functions that return stl-compatible iterators.


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Function, typename Arg, typename...is>
    concept col_result_is_lvalue = writable<Arg> and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg&&>(), 0))>& col, is...i) {
        {col} -> writable;
        requires requires { {f(col, i...)} -> std::same_as<void>; } or
          requires { {f(col, i...)} -> std::same_as<decltype(col)>; };
      };
#else
    template<typename Function, typename Arg, bool isvoid, typename = void, typename...is>
    struct col_result_is_valid : std::false_type {};

    template<typename Function, typename Arg, bool isvoid, typename...is>
    struct col_result_is_valid<Function, Arg, isvoid, std::enable_if_t<
      writable<decltype(column(std::declval<Arg&&>(), 0))> and std::is_same_v<
        typename std::invoke_result<Function, std::decay_t<decltype(column(std::declval<Arg&&>(), 0))>&, is...>::type,
        std::conditional_t<isvoid, void, std::decay_t<decltype(column(std::declval<Arg&&>(), 0))>&>>>, is...>
      : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool col_result_is_lvalue = writable<Arg> and std::is_lvalue_reference_v<Arg> and
      (col_result_is_valid<Function, Arg, true, void, is...>::value or
        col_result_is_valid<Function, Arg, false, void, is...>::value);


    template<typename Function, typename Arg, typename = void, typename...is>
    struct col_result_is_column_impl : std::false_type {};

    template<typename Function, typename Arg, typename...is>
    struct col_result_is_column_impl<Function, Arg, std::enable_if_t<
      column_vector<typename std::invoke_result<Function, decltype(column(std::declval<Arg&&>(), 0)), is...>::type>
      >, is...> : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool col_result_is_column_vector = col_result_is_column_impl<Function, Arg, void, is...>::value;
#endif


    template<bool index, typename F, typename Arg, std::size_t ... ints>
    inline decltype(auto) columnwise_impl(const F& f, Arg&& arg, std::index_sequence<ints...>)
    {
      static_assert(not dynamic_columns<Arg>);

      if constexpr ((index and detail::col_result_is_lvalue<F, Arg, std::size_t>) or
        (not index and detail::col_result_is_lvalue<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        return ([](const F& f, Arg&& arg) -> Arg& {
          decltype(auto) c {column<ints>(arg)};
          static_assert(writable<decltype(c)>);
          if constexpr (index) f(c, ints); else f(c);
          return arg;
        }(f, arg), ...);
      }
      else
      {
        if constexpr (index)
          return concatenate_horizontal(f(column<ints>(std::forward<Arg>(arg)), ints)...);
        else
          return concatenate_horizontal(f(column<ints>(std::forward<Arg>(arg)))...);
      }
    };


  } // namespace detail


#ifdef __cpp_concepts
  template<typename Function, eigen_matrix Arg> requires
    detail::col_result_is_lvalue<Function, Arg> or detail::col_result_is_lvalue<Function, Arg, std::size_t> or
    requires(const Function& f, Arg&& arg) {{f(column(std::forward<Arg>(arg), 0))} -> column_vector; } or
    requires(const Function& f, Arg&& arg, std::size_t i) {{f(column(std::forward<Arg>(arg), 0), i)} -> column_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (detail::col_result_is_lvalue<Function, Arg> or detail::col_result_is_lvalue<Function, Arg, std::size_t> or
    detail::col_result_is_column_vector<Function, Arg> or
    detail::col_result_is_column_vector<Function, Arg, std::size_t>), int> = 0>
#endif
  inline decltype(auto)
  apply_columnwise(const Function& f, Arg&& arg)
  {
    constexpr bool index = detail::col_result_is_lvalue<Function, Arg, std::size_t> or
#ifdef __cpp_concepts
      requires(const Function& f, Arg&& arg, std::size_t i) {
        {f(column(std::forward<Arg>(arg), 0), i)} -> column_vector;};
#else
      detail::col_result_is_column_vector<Function, Arg, std::size_t>;
#endif

    if constexpr (dynamic_columns<Arg>)
    {
      auto cols = get_dimensions_of<1>(arg);

      if constexpr ((index and detail::col_result_is_lvalue<Function, Arg, std::size_t>) or
        (not index and detail::col_result_is_lvalue<Function, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        for (std::size_t j = 0; j<cols; j++)
        {
          decltype(auto) c {column(arg, j)};
          static_assert(writable<decltype(c)>);
          if constexpr (index) f(c, j); else f(c);
        }
        return (arg);
      }
      else
      {
        auto res_col0 = [](const Function& f, Arg&& arg){
          auto col0 = column(std::forward<Arg>(arg), 0);
          if constexpr (index) return f(col0, 0); else return f(col0);
        }(f, std::forward<Arg>(arg));

        using ResultType = decltype(res_col0);
        using M = eigen_matrix_t<scalar_type_of_t<ResultType>, row_dimension_of_v<ResultType>, dynamic_size>;
        M m {get_dimensions_of<0>(res_col0), cols};

        column(m, 0) = res_col0;

        for (std::size_t j = 1; j<cols; j++)
        {
          if constexpr (index)
            column(m, j) = f(column(std::forward<Arg>(arg), j), j);
          else
            column(m, j) = f(column(std::forward<Arg>(arg), j));
        }
        return m;
      }
    }
    else
    {
      return detail::columnwise_impl<index>(
        f, std::forward<Arg>(arg), std::make_index_sequence<column_dimension_of_v<Arg>>());
    }
  }


  namespace detail
  {
    template<typename Function, std::size_t ... ints>
    inline auto cat_columnwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_horizontal(f(ints)...);
    };

#ifndef __cpp_concepts
    template<typename Function, typename = void, typename...is>
    struct columns_1_or_dynamic_impl : std::false_type {};

    template<typename Function, typename...is>
    struct columns_1_or_dynamic_impl<Function, std::enable_if_t<
      eigen_matrix<std::invoke_result_t<const Function&, is...>> and
      ( dynamic_columns<std::invoke_result_t<const Function&, is...>> or
        column_vector<std::invoke_result_t<const Function&, is...>>)>, is...> : std::true_type {};

    template<typename Function, typename...is>
    static constexpr bool columns_1_or_dynamic = columns_1_or_dynamic_impl<Function, void, is...>::value;
#endif
  }


#ifdef __cpp_concepts
  template<std::size_t...compile_time_columns, typename Function, std::convertible_to<std::size_t>...runtime_columns>
  requires (sizeof...(compile_time_columns) + sizeof...(runtime_columns) == 1) and
    ( requires(const Function& f) {
      {f()} -> eigen_matrix;
      requires requires { {f()} -> dynamic_columns; } or requires { {f()} -> column_vector; };
    } or requires(const Function& f, std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_columns; } or requires { {f(i)} -> column_vector; };
    })
#else
  template<std::size_t...compile_time_columns, typename Function, typename...runtime_columns, std::enable_if_t<
    (std::is_convertible_v<runtime_columns, std::size_t> and ...) and
    (sizeof...(compile_time_columns) + sizeof...(runtime_columns) == 1), int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, runtime_columns...c)
  {
#ifdef __cpp_concepts
    if constexpr (requires(std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_columns; } or requires { {f(i)} -> column_vector; }; })
#else
    static_assert(detail::columns_1_or_dynamic<Function> or detail::columns_1_or_dynamic<Function, std::size_t&>);

    if constexpr (detail::columns_1_or_dynamic<Function, std::size_t&>)
#endif
    {
      if constexpr (sizeof...(compile_time_columns) > 0)
      {
        return detail::cat_columnwise_impl(f, std::make_index_sequence<(compile_time_columns + ... + 0)>());
      }
      else
      {
        using R = std::invoke_result_t<const Function&, std::size_t>;
        using Scalar = scalar_type_of_t<R>;
        std::size_t cols = (c + ... + 0);

        if constexpr (dynamic_rows<R>)
        {
          auto r0 = f(0);
          auto rows = get_dimensions_of<0>(r0);
          eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          m.col(0) = r0;
          for (std::size_t j = 1; j < cols; j++)
          {
            auto rj = f(j);
            assert(get_dimensions_of<0>(rj) == rows);
            m.col(j) = rj;
          }
          return m;
        }
        else
        {
          constexpr auto rows = row_dimension_of_v<R>;
          eigen_matrix_t<Scalar, rows, dynamic_size> m {rows, cols};
          for (std::size_t j = 0; j < cols; j++)
          {
            m.col(j) = f(j);
          }
          return m;
        }
      }
    }
    else
    {
      auto r = f();
      using R = decltype(r);
      if constexpr (dynamic_columns<R>) assert (get_dimensions_of<1>(r) == 1);

      if constexpr (sizeof...(compile_time_columns) > 0)
        return make_self_contained(Eigen::Replicate<R, 1, compile_time_columns...>(r));
      else
        return make_self_contained(Eigen::Replicate<R, 1, Eigen::Dynamic>(r, 1, c...));
    }
  }


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Function, typename Arg, typename...is>
    concept row_result_is_lvalue = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(row(std::declval<Arg&&>(), 0))>& row, is...i) {
        {row} -> writable;
        requires requires { {f(row, i...)} -> std::same_as<void>; } or
          requires { {f(row, i...)} -> std::same_as<decltype(row)>; };
      };
#else
    template<typename Function, typename Arg, bool isvoid, typename = void, typename...is>
    struct row_result_is_valid : std::false_type {};

    template<typename Function, typename Arg, bool isvoid, typename...is>
    struct row_result_is_valid<Function, Arg, isvoid, std::enable_if_t<
      writable<decltype(row(std::declval<Arg&&>(), 0))> and std::is_same_v<
        typename std::invoke_result<Function, std::decay_t<decltype(row(std::declval<Arg&&>(), 0))>&, is...>::type,
        std::conditional_t<isvoid, void, std::decay_t<decltype(row(std::declval<Arg&&>(), 0))>&>>>, is...>
      : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool row_result_is_lvalue = writable<Arg> and std::is_lvalue_reference_v<Arg> and
      (row_result_is_valid<Function, Arg, true, void, is...>::value or
        row_result_is_valid<Function, Arg, false, void, is...>::value);


    template<typename Function, typename Arg, typename = void, typename...is>
    struct row_result_is_row_impl : std::false_type {};

    template<typename Function, typename Arg, typename...is>
    struct row_result_is_row_impl<Function, Arg, std::enable_if_t<
      row_vector<typename std::invoke_result<Function, decltype(row(std::declval<Arg&&>(), 0)), is...>::type>
      >, is...> : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool row_result_is_row_vector = row_result_is_row_impl<Function, Arg, void, is...>::value;
#endif


    template<bool index, typename F, typename Arg, std::size_t ... ints>
    inline decltype(auto) rowwise_impl(const F& f, Arg&& arg, std::index_sequence<ints...>)
    {
      if constexpr ((index and detail::row_result_is_lvalue<F, Arg, std::size_t>) or
        (not index and detail::row_result_is_lvalue<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        return ([](const F& f, Arg&& arg) -> Arg& {
          decltype(auto) r {row<ints>(arg)};
          static_assert(writable<decltype(r)>);
          if constexpr (index) f(r, ints); else f(r);
          return arg;
        }(f, arg), ...);
      }
      else
      {
        if constexpr (index)
          return concatenate_vertical(f(row<ints>(std::forward<Arg>(arg)), ints)...);
        else
          return concatenate_vertical(f(row<ints>(std::forward<Arg>(arg)))...);
      }
    };

  } // namespace detail


#ifdef __cpp_concepts
  template<typename Function, eigen_matrix Arg> requires
    detail::row_result_is_lvalue<Function, Arg> or detail::row_result_is_lvalue<Function, Arg, std::size_t> or
    requires(const Function& f, Arg&& arg) {{f(row(std::forward<Arg>(arg), 0))} -> row_vector; } or
    requires(const Function& f, Arg&& arg, std::size_t i) {{f(row(std::forward<Arg>(arg), 0), i)} -> row_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (detail::row_result_is_lvalue<Function, Arg> or detail::row_result_is_lvalue<Function, Arg, std::size_t> or
    detail::row_result_is_row_vector<Function, Arg> or
    detail::row_result_is_row_vector<Function, Arg, std::size_t>), int> = 0>
#endif
  inline decltype(auto)
  apply_rowwise(const Function& f, Arg&& arg)
  {
    constexpr bool index = detail::row_result_is_lvalue<Function, Arg, std::size_t> or
#ifdef __cpp_concepts
      requires(const Function& f, Arg&& arg, std::size_t i) {{f(row(std::forward<Arg>(arg), 0), i)} -> row_vector;};
#else
      detail::row_result_is_row_vector<Function, Arg, std::size_t>;
#endif

    if constexpr (dynamic_rows<Arg>)
    {
      auto rows = get_dimensions_of<0>(arg);

      if constexpr ((index and detail::row_result_is_lvalue<Function, Arg, std::size_t>) or
        (not index and detail::row_result_is_lvalue<Function, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        for (std::size_t i = 0; i<rows; i++)
        {
          decltype(auto) r {row(arg, i)};
          static_assert(writable<decltype(r)>);
          if constexpr (index) f(r, i); else f(r);
        }
        return (arg);
      }
      else
      {
        auto res_row0 = [](const Function& f, Arg&& arg){
          auto row0 = row(std::forward<Arg>(arg), 0);
          if constexpr (index) return f(row0, 0); else return f(row0);
        }(f, std::forward<Arg>(arg));

        using ResultType = decltype(res_row0);
        using M = eigen_matrix_t<
          scalar_type_of_t<ResultType>, dynamic_size, column_dimension_of_v<ResultType>>;
        M m {rows, get_dimensions_of<1>(res_row0)};

        row(m, 0) = res_row0;

        for (std::size_t i = 1; i<rows; i++)
        {
          if constexpr (index)
            row(m, i) = f(row(std::forward<Arg>(arg), i), i);
          else
            row(m, i) = f(row(std::forward<Arg>(arg), i));
        }
        return m;
      }
    }
    else
    {
      return detail::rowwise_impl<index>(
        f, std::forward<Arg>(arg), std::make_index_sequence<row_dimension_of_v<Arg>>());
    }
  }


  namespace detail
  {
    template<typename Function, std::size_t ... ints>
    inline auto cat_rowwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(f(ints)...);
    };

#ifndef __cpp_concepts
    template<typename Function, typename = void, typename...is>
    struct rows_1_or_dynamic_impl : std::false_type {};

    template<typename Function, typename...is>
    struct rows_1_or_dynamic_impl<Function, std::enable_if_t<
      eigen_matrix<std::invoke_result_t<const Function&, is...>> and
      ( dynamic_rows<std::invoke_result_t<const Function&, is...>> or
        row_vector<std::invoke_result_t<const Function&, is...>>)>, is...> : std::true_type {};

    template<typename Function, typename...is>
    static constexpr bool rows_1_or_dynamic = rows_1_or_dynamic_impl<Function, void, is...>::value;
#endif
  }


#ifdef __cpp_concepts
  template<std::size_t...compile_time_rows, typename Function, std::convertible_to<std::size_t>...runtime_rows> requires
    (sizeof...(compile_time_rows) + sizeof...(runtime_rows) == 1) and
    ( requires(const Function& f) {
      {f()} -> eigen_matrix;
      requires requires { {f()} -> dynamic_rows; } or requires { {f()} -> row_vector; };
    } or requires(const Function& f, std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_rows; } or requires { {f(i)} -> row_vector; };
    })
#else
  template<std::size_t...compile_time_rows, typename Function, typename...runtime_rows, std::enable_if_t<
    (std::is_convertible_v<runtime_rows, std::size_t> and ...) and
    (sizeof...(compile_time_rows) + sizeof...(runtime_rows) == 1), int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f, runtime_rows...r)
  {
#ifdef __cpp_concepts
    if constexpr (requires(std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_rows; } or requires { {f(i)} -> row_vector; }; })
#else
    static_assert(detail::rows_1_or_dynamic<Function> or detail::rows_1_or_dynamic<Function, std::size_t&>);

    if constexpr (detail::rows_1_or_dynamic<Function, std::size_t&>)
#endif
    {
      if constexpr (sizeof...(runtime_rows) == 0)
      {
        return detail::cat_rowwise_impl(f, std::make_index_sequence<(compile_time_rows + ... + 0)>());
      }
      else
      {
        using R = std::invoke_result_t<const Function&, std::size_t>;
        using Scalar = scalar_type_of_t<R>;
        std::size_t rows = (r + ... + 0);

        if constexpr (dynamic_columns<R>)
        {
          auto c0 = f(0);
          auto cols = get_dimensions_of<1>(c0);
          eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          m.row(0) = c0;
          for (std::size_t i = 1; i < rows; i++)
          {
            auto ci = f(i);
            assert(get_dimensions_of<1>(ci) == cols);
            m.row(i) = ci;
          }
          return m;
        }
        else
        {
          constexpr auto cols = column_dimension_of_v<R>;
          eigen_matrix_t<Scalar, dynamic_size, cols> m {rows, cols};
          for (std::size_t i = 0; i < rows; i++)
          {
            m.row(i) = f(i);
          }
          return m;
        }
      }
    }
    else
    {
      auto c = f();
      using C = decltype(c);
      if constexpr (dynamic_rows<C>) assert (get_dimensions_of<0>(c) == 1);

      if constexpr (sizeof...(compile_time_rows) > 0)
        return make_self_contained(Eigen::Replicate<C, compile_time_rows..., 1>(c));
      else
        return make_self_contained(Eigen::Replicate<C, Eigen::Dynamic, 1>(c, r..., 1));
    }
  }

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_INTERFACE_HPP
