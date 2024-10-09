/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
  template<indexible NestedObject, vector_space_descriptor Descriptor> requires 
    maybe_equivalent_to<vector_space_descriptor_of<NestedObject, 0>, Dimensions<euclidean_dimension_size_of_v<Descriptor>>>
#else
  template<typename NestedObject, typename Descriptor>
#endif
  struct FromEuclideanExpr : OpenKalman::internal::TypedMatrixBase<
    FromEuclideanExpr<NestedObject, Descriptor>, NestedObject, Descriptor>
  {

#ifndef __cpp_concepts
    static_assert(vector_space_descriptor<Descriptor>);
    static_assert(maybe_equivalent_to<vector_space_descriptor_of<NestedObject, 0>, Dimensions<euclidean_dimension_size_of_v<Descriptor>>>);
#endif

    using Scalar = scalar_type_of_t<NestedObject>;

  private:

    static constexpr auto columns = index_dimension_of_v<NestedObject, 1>; ///< Number of columns.

    using Base = OpenKalman::internal::TypedMatrixBase<FromEuclideanExpr, NestedObject, Descriptor>;

  public:

    using Base::Base;

    /**
     * Convert from a compatible from-euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor> and
      std::constructible_from<NestedObject, decltype(nested_object(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedObject {nested_object(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor> and
      std::is_constructible_v<NestedObject, decltype(nested_object(std::declval<Arg&&>()))>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& arg) : Base {nested_object(std::forward<Arg>(arg))} {}


    /**
     * Construct from a compatible to-euclidean expression.
     */
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor> and
      std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor> and
      std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& other) : Base {std::forward<Arg>(other)} {}


    /**
     * Construct from compatible matrix object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible matrix object and a \ref vector_space_descriptor object.
#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C> requires (not euclidean_expr<Arg>) and
      std::constructible_from<NestedObject, Arg&&> and
      (dynamic_vector_space_descriptor<C> or dynamic_vector_space_descriptor<Descriptor> or equivalent_to<C, Descriptor>)
#else
    template<typename Arg, typename C, std::enable_if_t<indexible<Arg> and vector_space_descriptor<C> and
      (not euclidean_expr<Arg>) and std::is_constructible_v<NestedObject, Arg&&> and
      (dynamic_vector_space_descriptor<C> or dynamic_vector_space_descriptor<Descriptor> or equivalent_to<C, Descriptor>), int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg, const Descriptor& c) : Base {std::forward<Arg>(arg), c} {}


#ifndef __cpp_concepts
    /**
     * /brief Construct from a list of coefficients.
     * /note If c++ concepts are available, this functionality is inherited from the base class.
     */
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == columns *
        (to_euclidean_expr<NestedObject> ? dimension_size_of_v<Descriptor> : euclidean_dimension_size_of_v<Descriptor>), int> = 0>
    FromEuclideanExpr(Args ... args) : Base {make_dense_object_from<NestedObject>(static_cast<const Scalar>(args)...)} {}
#endif


    /**
     * Assign from a compatible from-Euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      (equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>) and
      (index_dimension_of_v<Arg, 1> == columns) and std::assignable_from<std::add_lvalue_reference_t<NestedObject>, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero<NestedObject> and not identity_matrix<NestedObject>)
      {
        this->nested_object() = nested_object(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * Assign from a general Eigen matrix.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and
      (index_dimension_of_v<Arg, 0> == dimension_size_of_v<Descriptor>) and (index_dimension_of_v<Arg, 1> == columns) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_euclidean<Descriptor>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 0>::value == dimension_size_of_v<Descriptor>) and (index_dimension_of<Arg, 1>::value == columns) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_euclidean<Descriptor>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero<NestedObject> and not identity_matrix<NestedObject>)
      {
        this->nested_object() = to_euclidean<Descriptor>(std::forward<Arg>(arg));
      }
      return *this;
    }

  private:

    template<typename Arg>
    static auto to_euclidean_noalias(Arg&& arg)
    {
      if constexpr (euclidean_dimension_size_of_v<Descriptor> > dimension_size_of_v<Descriptor>)
        return to_dense_object(to_euclidean<Descriptor>(std::forward<Arg>(arg))); //< Prevent aliasing
      else
        return to_euclidean<Descriptor>(std::forward<Arg>(arg));
    }

  public:

    /// Increment from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (index_dimension_of_v<Arg, 1> == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (index_dimension_of<Arg, 1>::value == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>, int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      this->nested_object() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Increment from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and (index_dimension_of_v<Arg, 1> == columns) and
      (index_dimension_of_v<Arg, 0> == dimension_size_of_v<Descriptor>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      (index_dimension_of<Arg, 0>::value == dimension_size_of_v<Descriptor>), int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      this->nested_object() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Decrement from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (index_dimension_of_v<Arg, 1> == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (index_dimension_of<Arg, 1>::value == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      this->nested_object() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /// Decrement from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and (index_dimension_of_v<Arg, 1> == columns) and
      (index_dimension_of_v<Arg, 0> == dimension_size_of_v<Descriptor>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      (index_dimension_of<Arg, 0>::value == dimension_size_of_v<Descriptor>), int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      this->nested_object() = to_euclidean_noalias(*this - arg);
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
      this->nested_object() = to_euclidean_noalias(scalar_product(*this, scale));
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
      this->nested_object() = to_euclidean_noalias(scalar_quotient(*this, scale));
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
    template<typename Descriptor, typename NestedObject>
    struct indexible_object_traits<FromEuclideanExpr<Descriptor, NestedObject>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(nested_object(arg)); }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          if constexpr (n == 0_uz) return std::forward<Arg>(arg).my_dimension;
          else return OpenKalman::get_vector_space_descriptor(nested_object(std::forward<Arg>(arg)), n);
        }
        else
        {
          using Scalar = scalar_type_of<Arg>;
          if (n == 0) return DynamicDescriptor<Scalar> {std::forward<Arg>(arg).my_dimension};
          else return DynamicDescriptor<Scalar> {OpenKalman::get_vector_space_descriptor(nested_object(std::forward<Arg>(arg)), n)};
        }
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<NestedObject>)
          return constant_coefficient{arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<NestedObject>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<Qualification b>
      static constexpr bool one_dimensional = euclidean_vector_space_descriptor<Descriptor> and OpenKalman::one_dimensional<NestedObject, b>;


      template<TriangleType t>
      static constexpr bool is_triangular = euclidean_vector_space_descriptor<Descriptor> and triangular_matrix<NestedObject, t>;


      static constexpr bool is_triangular_adapter = false;


      static constexpr bool is_hermitian = hermitian_matrix<NestedObject> and euclidean_vector_space_descriptor<Descriptor>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_gettable<nested_object_of_t<Arg&&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<element_gettable<typename nested_object_of<Arg&&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr auto get(Arg&& arg, I i, Is...is)
      {
        if constexpr (has_untyped_index<Arg, 0>)
        {
          if constexpr (to_euclidean_expr<nested_object_of_t<Arg>>)
            return get_component(OpenKalman::nested_object(nested_object(std::forward<Arg>(arg))), i, is...);
          else
            return get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), i, is...);
        }
        else
        {
          auto g {[&arg, is...](std::size_t ix) { return get_component(nested_object(std::forward<Arg>(arg)), ix, is...); }};
          if constexpr (to_euclidean_expr<nested_object_of_t<Arg>>)
            return get_wrapped_component(get_vector_space_descriptor<0>(arg), g, i, 0);
          else
            return from_euclidean_element(get_vector_space_descriptor<0>(arg), g, i, 0);
        }
      }


      /**
       * \internal
       * \brief Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
       * \details This function sets the nested matrix, not the wrapped resulting matrix.
       * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
       * single-column vector is {-1., pi/2}, then set_component(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
       * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
       * \tparam Arg The matrix to set.
       * \tparam Scalar The value to set the coefficient to.
       * \param i The row of the coefficient.
       * \param j The column of the coefficient.
       */
  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_gettable<nested_object_of_t<Arg&>, 1 + sizeof...(Is)> and
        (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_object_of_t<Arg>>))
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<
        element_gettable<typename nested_object_of<Arg&>::type, 1 + sizeof...(Is)> and
        (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_object_of_t<Arg>>)), int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        if constexpr (has_untyped_index<Arg, 0>)
        {
          set_component(nested_object(nested_object(arg)), s, i, is...);
        }
        else if constexpr (to_euclidean_expr<nested_object_of_t<Arg>>)
        {
          auto s {[&arg, is...](const scalar_type_of_t<Arg>& x, std::size_t i) {
            return set_component(nested_object(nested_object(arg)), x, i, is...);
          }};
          auto g {[&arg, is...](std::size_t ix) {
            return get_component(nested_object(nested_object(arg)), ix, is...);
          }};
          set_wrapped_component(get_vector_space_descriptor<0>(arg), s, g, s, i, 0);
        }
        else
        {
          set_component(nested_object(arg), s, i, is...);
        }
      }


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires has_untyped_index<Arg, 0> and raw_data_defined_for<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<has_untyped_index<Arg, 0> and raw_data_defined_for<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(nested_object(arg)); }


      static constexpr Layout layout = euclidean_vector_space_descriptor<Descriptor> ? layout_of_v<NestedObject> : Layout::none;

    };

  } // namespace interface


} // OpenKalman


#endif //OPENKALMAN_FROMEUCLIDEANEXPR_HPP
