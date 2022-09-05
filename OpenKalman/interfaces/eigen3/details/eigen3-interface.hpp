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
      if constexpr ((Eigen::internal::traits<std::decay_t<T>>::Flags & Eigen::LvalueBit) != 0)
        return std::forward<Arg>(arg).coeffRef(static_cast<int>(i)...);
      else
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
    template<std::size_t index, typename Arg, typename...index_values>
    static constexpr decltype(auto) chip(Arg&& arg, index_values...is)
    {
      static_assert(index <= 1);

      if constexpr (native_eigen_matrix<Arg> or native_eigen_array<Arg>)
      {
        if constexpr (index == 1) return std::forward<Arg>(arg).col(static_cast<Eigen::Index>(is)...);
        else return std::forward<Arg>(arg).row(static_cast<Eigen::Index>(is)...);
      }
      else
      {
        chip<index>(make_dense_writable_matrix_from(std::forward<Arg>(arg)), is...);
      }
    }

  private:

    // Concatenate one or more Eigen::MatrixBase objects diagonally.
    template<typename M, typename ... Vs, std::size_t ... ints>
    static void concatenate_diagonal_impl(M& m, const std::tuple<Vs...>& vs_tup, std::index_sequence<ints...>)
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

  public:

    template<std::size_t...indices, typename V, typename...Vs>
    static constexpr decltype(auto) concatenate(V&& v, Vs&&...vs)
    {
      using Scalar = std::common_type_t<scalar_type_of_t<V>, scalar_type_of_t<Vs>...>;

      if constexpr (sizeof...(indices) == 1 and ((indices == 0) and ...)) // vertical
      {
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
      else if constexpr (sizeof...(indices) == 1 and ((indices == 1) and ...)) // horizontal
      {
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
      else // diagonal
      {
        static_assert(sizeof...(indices) == 2 and ((indices <= 1) and ...));
        auto seq = std::make_index_sequence<(sizeof...(vs) + 1) * (sizeof...(vs) + 1) - 1> {};

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
          {
            auto rows = (get_dimensions_of<0>(v) + ... + get_dimensions_of<0>(vs));
            auto cols = (get_dimensions_of<1>(v) + ... + get_dimensions_of<1>(vs));
            eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
            concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
            return m;
          }
          else
          {
            auto rows = (get_dimensions_of<0>(v) + ... + get_dimensions_of<0>(vs));
            constexpr auto cols = (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>);
            eigen_matrix_t<Scalar, dynamic_size, cols> m {rows, cols};
            concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
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
            concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
            return m;
          }
          else
          {
            constexpr auto rows = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
            constexpr auto cols = (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>);
            eigen_matrix_t<Scalar, rows, cols> m;
            concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
            return m;
          }
        }
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
        std::make_index_sequence<sizeof...(Ds)> {});
    }

  public:

#ifdef __cpp_concepts
    template<typename...Ds, typename Operation, typename...Args> requires (sizeof...(Args) <= 3) and
      std::invocable<Operation&&, scalar_type_of_t<Args>...> and
      scalar_type<std::invoke_result_t<Operation&&, scalar_type_of_t<Args>...>>
#else
    template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<(sizeof...(Args) <= 3) and
      std::is_invocable<Operation&&, typename scalar_type_of<Args>::type...>::value and
      scalar_type<typename std::invoke_result<Operation&&, typename scalar_type_of<Args>::type...>::type>, int> = 0>
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


#endif //OPENKALMAN_EIGEN3_INTERFACE_HPP
