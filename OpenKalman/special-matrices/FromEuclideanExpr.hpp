/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief FromEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_FROMEUCLIDEANEXPR_HPP
#define OPENKALMAN_FROMEUCLIDEANEXPR_HPP

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor TypedIndex, typename NestedMatrix>
  requires (dynamic_vector_space_descriptor<TypedIndex> == dynamic_dimension<NestedMatrix, 0>) and
    (not fixed_vector_space_descriptor<TypedIndex> or euclidean_dimension_size_of_v<TypedIndex> == index_dimension_of_v<NestedMatrix, 0>) and
    (not dynamic_vector_space_descriptor<TypedIndex> or
      std::same_as<typename TypedIndex::Scalar, scalar_type_of_t<NestedMatrix>>)
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct FromEuclideanExpr : OpenKalman::internal::TypedMatrixBase<
    FromEuclideanExpr<TypedIndex, NestedMatrix>, NestedMatrix, TypedIndex>
  {

#ifndef __cpp_concepts
    static_assert(fixed_vector_space_descriptor<TypedIndex>);
    static_assert(dynamic_vector_space_descriptor<TypedIndex> == dynamic_dimension<NestedMatrix, 0>);
    static_assert(not fixed_vector_space_descriptor<TypedIndex> or euclidean_dimension_size_of_v<TypedIndex> == index_dimension_of_v<NestedMatrix, 0>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>;

  private:

    static constexpr auto columns = index_dimension_of_v<NestedMatrix, 1>; ///< Number of columns.

    using Base = OpenKalman::internal::TypedMatrixBase<FromEuclideanExpr, NestedMatrix, TypedIndex>;

  public:

    using Base::Base;

    /**
     * Convert from a compatible from-euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex> and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /**
     * Construct from a compatible to-euclidean expression.
     */
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex> and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& other) noexcept : Base {std::forward<Arg>(other)} {}


    /**
     * Construct from compatible matrix object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible matrix object and a \ref vector_space_descriptor object.
#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C> requires (not euclidean_expr<Arg>) and
      std::constructible_from<NestedMatrix, Arg&&> and
      (dynamic_vector_space_descriptor<C> or dynamic_vector_space_descriptor<TypedIndex> or equivalent_to<C, TypedIndex>)
#else
    template<typename Arg, typename C, std::enable_if_t<indexible<Arg> and vector_space_descriptor<C> and
      (not euclidean_expr<Arg>) and std::is_constructible_v<NestedMatrix, Arg&&> and
      (dynamic_vector_space_descriptor<C> or dynamic_vector_space_descriptor<TypedIndex> or equivalent_to<C, TypedIndex>), int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg, const TypedIndex& c) noexcept : Base {std::forward<Arg>(arg), c} {}


#ifndef __cpp_concepts
    /**
     * /brief Construct from a list of coefficients.
     * /note If c++ concepts are available, this functionality is inherited from the base class.
     */
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == columns *
        (to_euclidean_expr<NestedMatrix> ? dimension_size_of_v<TypedIndex> : euclidean_dimension_size_of_v<TypedIndex>), int> = 0>
    FromEuclideanExpr(Args ... args) : Base {make_dense_writable_matrix_from<NestedMatrix>(static_cast<const Scalar>(args)...)} {}
#endif


    /**
     * Assign from a compatible from-Euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      (equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex>) and
      (index_dimension_of_v<Arg, 1> == columns) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * Assign from a general Eigen matrix.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and
      (index_dimension_of_v<Arg, 0> == dimension_size_of_v<TypedIndex>) and (index_dimension_of_v<Arg, 1> == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<TypedIndex>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 0>::value == dimension_size_of_v<TypedIndex>) and (index_dimension_of<Arg, 1>::value == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<TypedIndex>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = to_euclidean<TypedIndex>(std::forward<Arg>(arg));
      }
      return *this;
    }

  private:

    template<typename Arg>
    static auto to_euclidean_noalias(Arg&& arg)
    {
      if constexpr (euclidean_dimension_size_of_v<TypedIndex> > dimension_size_of_v<TypedIndex>)
        return make_dense_writable_matrix_from(to_euclidean<TypedIndex>(std::forward<Arg>(arg))); //< Prevent aliasing
      else
        return to_euclidean<TypedIndex>(make_self_contained<Arg>(std::forward<Arg>(arg)));
    }

  public:

    /// Increment from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (index_dimension_of_v<Arg, 1> == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (index_dimension_of<Arg, 1>::value == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Increment from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and (index_dimension_of_v<Arg, 1> == columns) and
      (index_dimension_of_v<Arg, 0> == dimension_size_of_v<TypedIndex>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      (index_dimension_of<Arg, 0>::value == dimension_size_of_v<TypedIndex>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Decrement from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (index_dimension_of_v<Arg, 1> == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (index_dimension_of<Arg, 1>::value == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, TypedIndex>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /// Decrement from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and (index_dimension_of_v<Arg, 1> == columns) and
      (index_dimension_of_v<Arg, 0> == dimension_size_of_v<TypedIndex>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      (index_dimension_of<Arg, 0>::value == dimension_size_of_v<TypedIndex>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /**
     * Multiply by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S scale)
    {
      this->nested_matrix() = to_euclidean_noalias(*this * scale);
      return *this;
    }


    /**
     * Divide by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S scale)
    {
      this->nested_matrix() = to_euclidean_noalias(*this / scale);
      return *this;
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<indexible Arg, vector_space_descriptor C>
#else
  template<typename Arg, typename C, std::enable_if_t<indexible<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
  FromEuclideanExpr(Arg&&, const C&) -> FromEuclideanExpr<C, passable_t<Arg>>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename TypedIndex, typename NestedMatrix>
    struct indexible_object_traits<FromEuclideanExpr<TypedIndex, NestedMatrix>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg>
      static constexpr auto get_index_count(const Arg& arg) { return OpenKalman::get_index_count(nested_matrix(arg)); }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          if constexpr (static_index_value_of_v<N> == 0) return std::forward<Arg>(arg).my_dimension;
          else return OpenKalman::get_vector_space_descriptor(nested_matrix(std::forward<Arg>(arg)), n);
        }
        else
        {
          using Scalar = scalar_type_of<Arg>;
          if (n == 0) return DynamicTypedIndex<Scalar> {std::forward<Arg>(arg).my_dimension};
          else return DynamicTypedIndex<Scalar> {OpenKalman::get_vector_space_descriptor(nested_matrix(std::forward<Arg>(arg)), n)};
        }
      }

      using type = std::tuple<NestedMatrix>;

      static constexpr bool has_runtime_parameters = false;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix(std::forward<Arg>(arg)));
        return ToEuclideanExpr<TypedIndex, decltype(n)> {std::move(n)};
      }

      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<NestedMatrix>)
          return constant_coefficient{arg.nestedExpression()};
        else
          return std::monostate {};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<NestedMatrix>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = euclidean_vector_space_descriptor<TypedIndex> and one_by_one_matrix<NestedMatrix, b>;

      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = euclidean_vector_space_descriptor<TypedIndex> and triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_hermitian = hermitian_matrix<NestedMatrix> and euclidean_vector_space_descriptor<TypedIndex>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_gettable<nested_matrix_of_t<Arg&&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg&&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr auto get(Arg&& arg, I i, Is...is)
      {
        if constexpr (has_untyped_index<Arg, 0>)
        {
          if constexpr (to_euclidean_expr<nested_matrix_of_t<Arg>>)
            return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), i, is...);
          else
            return get_element(nested_matrix(std::forward<Arg>(arg)), i, is...);
        }
        else
        {
          auto g {[&arg, is...](std::size_t ix) { return get_element(nested_matrix(std::forward<Arg>(arg)), ix, is...); }};
          if constexpr (to_euclidean_expr<nested_matrix_of_t<Arg>>)
            return wrap_get_element(get_vector_space_descriptor<0>(arg), g, i, 0);
          else
            return from_euclidean_element(get_vector_space_descriptor<0>(arg), g, i, 0);
        }
      }


      /**
       * \internal
       * \brief Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
       * \details This function sets the nested matrix, not the wrapped resulting matrix.
       * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
       * single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
       * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
       * \tparam Arg The matrix to set.
       * \tparam Scalar The value to set the coefficient to.
       * \param i The row of the coefficient.
       * \param j The column of the coefficient.
       */
  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_gettable<nested_matrix_of_t<Arg&>, 1 + sizeof...(Is)> and
        (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>))
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<
        element_gettable<typename nested_matrix_of<Arg&>::type, 1 + sizeof...(Is)> and
        (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>)), int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        if constexpr (has_untyped_index<Arg, 0>)
        {
          set_element(nested_matrix(nested_matrix(arg)), s, i, is...);
        }
        else if constexpr (to_euclidean_expr<nested_matrix_of_t<Arg>>)
        {
          auto s {[&arg, is...](const scalar_type_of_t<Arg>& x, std::size_t i) {
            return set_element(nested_matrix(nested_matrix(arg)), x, i, is...);
          }};
          auto g {[&arg, is...](std::size_t ix) {
            return get_element(nested_matrix(nested_matrix(arg)), ix, is...);
          }};
          wrap_set_element(get_vector_space_descriptor<0>(arg), s, g, s, i, 0);
        }
        else
        {
          set_element(nested_matrix(arg), s, i, is...);
        }
      }


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires has_untyped_index<Arg, 0> and directly_accessible<nested_matrix_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<has_untyped_index<Arg, 0> and directly_accessible<typename nested_matrix_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto*
      data(Arg& arg) { return internal::raw_data(nested_matrix(arg)); }


      static constexpr Layout layout = euclidean_vector_space_descriptor<TypedIndex> ? layout_of_v<NestedMatrix> : Layout::none;

    };

  } // namespace interface


} // OpenKalman


#endif //OPENKALMAN_FROMEUCLIDEANEXPR_HPP
